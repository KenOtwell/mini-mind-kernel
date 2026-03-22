"""
End-to-end integration tests for the /ingest pipeline.

Sends TickPackages through the FastAPI app with a mocked LLM backend.
Verifies the full flow: accumulate → schedule → prompt → LLM → expand → respond.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from progeny.api.server import app
from progeny.api import routes
from progeny.src.event_accumulator import EventAccumulator
from progeny.src.agent_scheduler import AgentScheduler
from progeny.src.llm_client import GenerateResult
from tests.fixtures.factories import make_turn_package, make_data_package


@pytest.fixture(autouse=True)
def _fresh_state():
    """Reset pipeline state between tests."""
    routes._accumulator = EventAccumulator()
    routes._scheduler = AgentScheduler()
    yield


def _mock_llm_response(agents: list[str]) -> str:
    """Build a valid LLM response JSON for the given agents."""
    return json.dumps({
        "responses": [
            {
                "agent_id": agent_id,
                "utterance": f"Hello from {agent_id}.",
                "actor_value_deltas": {"Mood": 3},
            }
            for agent_id in agents
        ]
    })


class TestDataOnlyTick:
    @pytest.mark.asyncio
    async def test_data_tick_returns_ack(self):
        pkg = make_data_package()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ingest",
                content=pkg.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accumulated"
        assert "tick_id" in data


class TestTurnTrigger:
    @pytest.mark.asyncio
    async def test_turn_returns_turn_response(self):
        pkg = make_turn_package("What do you think?", active_npc_ids=["Lydia"])
        mock_response = _mock_llm_response(["Lydia"])

        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerateResult(content=mock_response)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert "responses" in data
        assert len(data["responses"]) == 1
        assert data["responses"][0]["agent_id"] == "Lydia"
        assert data["responses"][0]["utterance"] == "Hello from Lydia."
        assert data["model_used"] == "llama.cpp"
        assert data["processing_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_multi_npc_turn(self):
        pkg = make_turn_package(
            "Hello everyone!",
            active_npc_ids=["Lydia", "Belethor", "Ysolda"],
        )
        mock_response = _mock_llm_response(["Lydia", "Belethor", "Ysolda"])

        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerateResult(content=mock_response)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        data = resp.json()
        assert len(data["responses"]) == 3
        ids = [r["agent_id"] for r in data["responses"]]
        assert ids == ["Lydia", "Belethor", "Ysolda"]

    @pytest.mark.asyncio
    async def test_llm_error_returns_graceful_degradation(self):
        """LLM failure → graceful degradation with empty agent responses."""
        from progeny.src.llm_client import LLMError

        pkg = make_turn_package("Help!", active_npc_ids=["Lydia"])

        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = LLMError("Connection failed")
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_used"] == "error"
        # Graceful degradation: agent entries exist but have no utterances
        assert len(data["responses"]) == 1
        assert data["responses"][0]["agent_id"] == "Lydia"
        assert data["responses"][0]["utterance"] is None

    @pytest.mark.asyncio
    async def test_no_active_npcs_returns_empty(self):
        """Turn trigger with no NPCs → no LLM call, empty responses."""
        pkg = make_turn_package("Anyone there?", active_npc_ids=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ingest",
                content=pkg.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )

        data = resp.json()
        assert data["responses"] == []
        assert data["model_used"] == "none"


class TestPromptPassedToLLM:
    @pytest.mark.asyncio
    async def test_prompt_contains_player_input(self):
        """Verify the LLM receives the player's text in the prompt."""
        pkg = make_turn_package("Tell me about dragons", active_npc_ids=["Lydia"])
        mock_response = _mock_llm_response(["Lydia"])
        captured_messages = []

        async def capture_generate(messages):
            captured_messages.extend(messages)
            return GenerateResult(content=mock_response)

        with patch("progeny.src.llm_client.generate", side_effect=capture_generate):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        assert len(captured_messages) == 2
        json_part = captured_messages[1]["content"].split("\n\n")[0]
        data_msg = json.loads(json_part)
        assert data_msg["player_input"]["text"] == "Tell me about dragons"

    @pytest.mark.asyncio
    async def test_prompt_contains_agent_blocks(self):
        """With parallel dispatch, each agent gets its own prompt."""
        pkg = make_turn_package("Hi", active_npc_ids=["Lydia", "Belethor"])
        captured_calls: list[list[dict]] = []

        async def capture_generate(messages):
            captured_calls.append(list(messages))
            # Return response for whichever agent is in this group
            json_part = messages[1]["content"].split("\n\n")[0]
            data = json.loads(json_part)
            agent_ids = [a["agent_id"] for a in data["agents"]]
            return GenerateResult(content=_mock_llm_response(agent_ids))

        with patch("progeny.src.llm_client.generate", side_effect=capture_generate):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # Each Tier 0 agent gets its own dispatch group
        assert len(captured_calls) == 2
        # Collect all agents across all groups
        all_agent_ids = []
        all_present_npcs = []
        for call in captured_calls:
            json_part = call[1]["content"].split("\n\n")[0]
            data = json.loads(json_part)
            all_agent_ids.extend(a["agent_id"] for a in data["agents"])
            all_present_npcs.append(data.get("present_npcs", []))
        assert sorted(all_agent_ids) == ["Belethor", "Lydia"]
        # Each group sees ALL present NPCs for cross-agent awareness
        for present in all_present_npcs:
            assert "Lydia" in present
            assert "Belethor" in present
        # Final merged response has both
        data = resp.json()
        resp_ids = [r["agent_id"] for r in data["responses"]]
        assert "Lydia" in resp_ids
        assert "Belethor" in resp_ids


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/health",
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "llm_connected" in data
