"""
Tests for falcon.src.event_parsers — structural decoders per SKSE event type.

Covers: happy path, missing/optional field defaults, and malformed input
returning None without raising. parse_typed_data dispatcher is tested last.
"""
import pytest

from falcon.src.event_parsers import (
    parse_speech,
    parse_addnpc,
    parse_updatestats,
    parse_quest_json,
    parse_quest_update,
    parse_itemtransfer,
    parse_typed_data,
)


# ---------------------------------------------------------------------------
# _speech → SpeechData
# ---------------------------------------------------------------------------

class TestParseSpeech:
    def test_happy_path(self):
        data = (
            '{"listener": "Lydia", "speaker": "Player", "speech": "Hello",'
            ' "location": "Whiterun", "companions": ["Faendal"], "distance": 5.0}'
        )
        result = parse_speech(data)
        assert result is not None
        assert result["listener"] == "Lydia"
        assert result["speaker"] == "Player"
        assert result["speech"] == "Hello"
        assert result["location"] == "Whiterun"
        assert result["companions"] == ["Faendal"]
        assert result["distance"] == 5.0

    def test_missing_optional_fields_use_defaults(self):
        data = '{"listener": "Lydia", "speaker": "Player", "speech": "Hi", "location": "Whiterun"}'
        result = parse_speech(data)
        assert result is not None
        assert result["companions"] == []
        assert result["distance"] == 0.0

    def test_invalid_distance_defaults_to_zero(self):
        data = '{"listener": "A", "speaker": "B", "speech": "C", "location": "D", "distance": "far"}'
        result = parse_speech(data)
        assert result is not None
        assert result["distance"] == 0.0

    def test_malformed_json_returns_none(self):
        assert parse_speech("not json") is None

    def test_non_object_json_returns_none(self):
        assert parse_speech("[1, 2, 3]") is None

    def test_empty_string_returns_none(self):
        assert parse_speech("") is None


# ---------------------------------------------------------------------------
# addnpc → NpcRegistration
# ---------------------------------------------------------------------------

# Minimal valid addnpc: name@base@gender@race@refid + 18 skill slots (empty)
# + 10 equipment slots (empty) + 8 stat fields + mods + factions + class
_ADDNPC_FULL = (
    "Lydia@LydiaBase@Female@Nord@00012345"
    "@50@55@60@45@30@40@35@25@20@70@65@10@15@80@50@45@55@30"  # 18 skills
    "@@@@@@@@@@"                                                # 10 equipment slots
    "@25@100.0@100.0@50.0@50.0@80.0@80.0@1.0"                 # 8 stats
    "@SkyrimMod.esp#OtherMod.esp"                              # mods
    "@000A1A2B:1#000C3D4E:0"                                   # factions
    "@Knight:00015C4A"                                          # class
)


class TestParseAddnpc:
    def test_happy_path(self):
        result = parse_addnpc(_ADDNPC_FULL)
        assert result is not None
        assert result["name"] == "Lydia"
        assert result["base"] == "LydiaBase"
        assert result["gender"] == "Female"
        assert result["race"] == "Nord"
        assert result["refid"] == "00012345"

    def test_skills_extracted(self):
        result = parse_addnpc(_ADDNPC_FULL)
        assert result is not None
        assert "archery" in result["skills"]
        assert result["skills"]["archery"] == "50"

    def test_stats_extracted(self):
        result = parse_addnpc(_ADDNPC_FULL)
        assert result is not None
        assert result["stats"]["level"] == 25
        assert result["stats"]["health"] == 100.0
        assert result["stats"]["scale"] == 1.0

    def test_mods_parsed(self):
        result = parse_addnpc(_ADDNPC_FULL)
        assert result is not None
        assert "SkyrimMod.esp" in result["mods"]
        assert "OtherMod.esp" in result["mods"]

    def test_factions_parsed(self):
        result = parse_addnpc(_ADDNPC_FULL)
        assert result is not None
        assert len(result["factions"]) == 2
        assert result["factions"][0]["formid"] == "000A1A2B"
        assert result["factions"][0]["rank"] == 1

    def test_class_parsed(self):
        result = parse_addnpc(_ADDNPC_FULL)
        assert result is not None
        assert result["class_info"] is not None
        assert result["class_info"]["name"] == "Knight"

    def test_missing_name_returns_none(self):
        assert parse_addnpc("@base@female@nord") is None

    def test_empty_string_returns_none(self):
        assert parse_addnpc("") is None

    def test_partial_data_uses_stat_defaults(self):
        result = parse_addnpc("MinimalNPC")
        assert result is not None
        assert result["name"] == "MinimalNPC"
        assert result["stats"]["level"] == 1
        assert result["stats"]["health"] == 0.0
        assert result["stats"]["scale"] == 1.0
        assert result["mods"] == []
        assert result["factions"] == []
        assert result["class_info"] is None


# ---------------------------------------------------------------------------
# updatestats → NpcStats
# ---------------------------------------------------------------------------

class TestParseUpdatestats:
    def test_happy_path(self):
        result = parse_updatestats("Lydia@25@95.0@100.0@40.0@50.0@70.0@80.0@1.0")
        assert result is not None
        assert result["npc_name"] == "Lydia"
        assert result["level"] == 25
        assert result["health"] == 95.0
        assert result["health_max"] == 100.0
        assert result["magicka"] == 40.0
        assert result["stamina_max"] == 80.0
        assert result["scale"] == 1.0

    def test_missing_name_returns_none(self):
        assert parse_updatestats("@25@100.0@100.0") is None

    def test_empty_string_returns_none(self):
        assert parse_updatestats("") is None

    def test_partial_data_uses_defaults(self):
        result = parse_updatestats("Lydia@25")
        assert result is not None
        assert result["level"] == 25
        assert result["health"] == 0.0
        assert result["scale"] == 1.0

    def test_invalid_float_uses_default(self):
        result = parse_updatestats("Lydia@25@notaFloat@100.0")
        assert result is not None
        assert result["health"] == 0.0
        assert result["health_max"] == 100.0


# ---------------------------------------------------------------------------
# _quest → QuestData
# ---------------------------------------------------------------------------

class TestParseQuestJson:
    def test_happy_path(self):
        data = (
            '{"formId": "MQ101", "name": "Unbound", "currentbrief": "Escape Helgen",'
            ' "stage": "10", "data": {"questgiver": "Hadvar"}, "status": "active"}'
        )
        result = parse_quest_json(data)
        assert result is not None
        assert result["form_id"] == "MQ101"
        assert result["name"] == "Unbound"
        assert result["brief"] == "Escape Helgen"
        assert result["giver"] == "Hadvar"
        assert result["status"] == "active"

    def test_missing_optional_fields_use_defaults(self):
        result = parse_quest_json('{"formId": "MQ101"}')
        assert result is not None
        assert result["name"] == ""
        assert result["giver"] == ""
        assert result["status"] == ""

    def test_malformed_json_returns_none(self):
        assert parse_quest_json("not json") is None

    def test_non_object_returns_none(self):
        assert parse_quest_json('"just a string"') is None

    def test_empty_string_returns_none(self):
        assert parse_quest_json("") is None


# ---------------------------------------------------------------------------
# _uquest / _questdata → QuestUpdate
# ---------------------------------------------------------------------------

class TestParseQuestUpdate:
    def test_happy_path(self):
        result = parse_quest_update("MQ101@somedata@Escape Helgen@10")
        assert result is not None
        assert result["form_id"] == "MQ101"
        assert result["briefing"] == "Escape Helgen"
        assert result["stage"] == "10"

    def test_empty_returns_none(self):
        assert parse_quest_update("") is None

    def test_no_briefing_uses_default(self):
        result = parse_quest_update("MQ101")
        assert result is not None
        assert result["briefing"] == ""
        assert result["stage"] == ""

    def test_partial_with_two_fields(self):
        result = parse_quest_update("MQ101@extra")
        assert result is not None
        assert result["form_id"] == "MQ101"
        assert result["briefing"] == ""


# ---------------------------------------------------------------------------
# itemtransfer → ItemTransfer
# ---------------------------------------------------------------------------

class TestParseItemtransfer:
    def test_happy_path(self):
        result = parse_itemtransfer("Lydia gave 2 Health Potion to Faendal")
        assert result is not None
        assert result["source"] == "Lydia"
        assert result["count"] == 2
        assert result["item_name"] == "Health Potion"
        assert result["dest"] == "Faendal"

    def test_singular_item(self):
        result = parse_itemtransfer("Player gave 1 Iron Sword to Lydia")
        assert result is not None
        assert result["count"] == 1
        assert result["item_name"] == "Iron Sword"

    def test_case_insensitive(self):
        result = parse_itemtransfer("Faendal GAVE 3 Arrow to Player")
        assert result is not None
        assert result["source"] == "Faendal"
        assert result["count"] == 3

    def test_malformed_returns_none(self):
        assert parse_itemtransfer("not a transfer") is None

    def test_empty_returns_none(self):
        assert parse_itemtransfer("") is None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestParseTypedData:
    def test_speech_dispatches(self):
        result = parse_typed_data(
            "_speech",
            '{"listener": "Lydia", "speaker": "Player", "speech": "Hi", "location": "Whiterun"}',
        )
        assert result is not None
        assert result["listener"] == "Lydia"

    def test_addnpc_dispatches(self):
        result = parse_typed_data("addnpc", "Lydia")
        assert result is not None
        assert result["name"] == "Lydia"

    def test_updatestats_dispatches(self):
        result = parse_typed_data("updatestats", "Faendal@10@80.0@80.0@0@0@60@60@1.0")
        assert result is not None
        assert result["npc_name"] == "Faendal"

    def test_quest_json_dispatches(self):
        result = parse_typed_data("_quest", '{"formId": "BQ01"}')
        assert result is not None
        assert result["form_id"] == "BQ01"

    def test_uquest_dispatches(self):
        result = parse_typed_data("_uquest", "BQ01@x@Take back the hold@20")
        assert result is not None
        assert result["briefing"] == "Take back the hold"

    def test_questdata_dispatches(self):
        result = parse_typed_data("_questdata", "BQ01@x@Some briefing@5")
        assert result is not None
        assert result["form_id"] == "BQ01"

    def test_itemtransfer_dispatches(self):
        result = parse_typed_data("itemtransfer", "Player gave 1 Sword to Lydia")
        assert result is not None
        assert result["dest"] == "Lydia"

    def test_unknown_type_returns_none(self):
        assert parse_typed_data("info", "Lydia drew her weapon") is None
        assert parse_typed_data("death", "Bandit died") is None
        assert parse_typed_data("location", "Whiterun") is None

    def test_malformed_input_returns_none_not_raises(self):
        # Must not raise even with garbage input for known types
        result = parse_typed_data("_speech", "not json at all {{{")
        assert result is None

    def test_addnpc_malformed_returns_none(self):
        result = parse_typed_data("addnpc", "")
        assert result is None
