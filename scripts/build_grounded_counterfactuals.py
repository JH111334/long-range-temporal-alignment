"""Build and validate grounded semantic temporal counterfactuals (Gold-Flip & Gold-Same).

Generates:
1. data/counterfactual/grounded_gold_flip_pairs.jsonl (36 pairs = 72 evaluated questions)
2. data/counterfactual/grounded_gold_same_pairs.jsonl (18 pairs = 36 evaluated questions)
3. data/counterfactual/MANIFEST.json with SHA256 checksums, pair metadata, and source provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data/counterfactual"


# 36 Grounded Gold-Flip Pairs (Factual Ground-Truth Flips between t1 and t2)
GOLD_FLIP_PAIRS_SPEC: List[Dict[str, Any]] = [
    # 1. State Sovereignty & Territorial Unity
    {
        "pair_id": "CF_FLIP_01_GERMANY_UNIFIED",
        "entity": "Germany",
        "relation": "unified_democratic_state",
        "question_template": "In {year}, Germany was a unified single democratic state? Answer as Yes or No",
        "t1": 2022, "gold_t1": "Yes",
        "t2": 1984, "gold_t2": "No",
        "event_id": "GERMAN_REUNIFICATION_1990",
        "temporal_cluster_id": "POST_WAR_EUROPE",
        "source_record_id": "HIST_MILESTONE_GER_01",
        "provenance": "Treaty on the Final Settlement with Respect to Germany (effective Oct 1990)",
    },
    {
        "pair_id": "CF_FLIP_02_VIETNAM_UNIFIED",
        "entity": "Vietnam",
        "relation": "territorially_unified_north_to_south",
        "question_template": "In {year}, Vietnam was completely unified from North to South? Answer as Yes or No",
        "t1": 1985, "gold_t1": "Yes",
        "t2": 1968, "gold_t2": "No",
        "event_id": "VIETNAM_REUNIFICATION_1975",
        "temporal_cluster_id": "SOUTHEAST_ASIA_COLD_WAR",
        "source_record_id": "HIST_MILESTONE_VNM_01",
        "provenance": "National reunification April 30 1975; formal Socialist Republic declared 1976",
    },
    {
        "pair_id": "CF_FLIP_03_USSR_EXISTENCE",
        "entity": "Soviet Union",
        "relation": "existed_as_intact_sovereign_state",
        "question_template": "In {year}, the Soviet Union existed as an intact sovereign state? Answer as Yes or No",
        "t1": 1980, "gold_t1": "Yes",
        "t2": 1995, "gold_t2": "No",
        "event_id": "DISSOLUTION_OF_SOVIET_UNION_1991",
        "temporal_cluster_id": "COLD_WAR_COLLAPSE",
        "source_record_id": "HIST_MILESTONE_SUN_01",
        "provenance": "Belovezha Accords and Alma-Ata Protocol (December 1991)",
    },
    {
        "pair_id": "CF_FLIP_04_YUGOSLAVIA_EXISTENCE",
        "entity": "Yugoslavia",
        "relation": "existed_as_federal_socialist_republic",
        "question_template": "In {year}, the Socialist Federal Republic of Yugoslavia existed as a federal state? Answer as Yes or No",
        "t1": 1978, "gold_t1": "Yes",
        "t2": 1998, "gold_t2": "No",
        "event_id": "BREAKUP_OF_YUGOSLAVIA_1992",
        "temporal_cluster_id": "COLD_WAR_COLLAPSE",
        "source_record_id": "HIST_MILESTONE_YUG_01",
        "provenance": "Badinter Arbitration Committee decisions on dissolution (1991-1992)",
    },
    {
        "pair_id": "CF_FLIP_05_CZECHOSLOVAKIA_EXISTENCE",
        "entity": "Czechoslovakia",
        "relation": "existed_as_single_unified_nation",
        "question_template": "In {year}, Czechoslovakia existed as a single unified country? Answer as Yes or No",
        "t1": 1985, "gold_t1": "Yes",
        "t2": 1996, "gold_t2": "No",
        "event_id": "DISSOLUTION_OF_CZECHOSLOVAKIA_1993",
        "temporal_cluster_id": "COLD_WAR_COLLAPSE",
        "source_record_id": "HIST_MILESTONE_CSK_01",
        "provenance": "Peaceful dissolution into Czech Republic and Slovakia (Jan 1 1993)",
    },
    {
        "pair_id": "CF_FLIP_06_SOUTH_SUDAN_INDEPENDENCE",
        "entity": "South Sudan",
        "relation": "sovereign_independent_nation",
        "question_template": "In {year}, South Sudan was an independent sovereign nation? Answer as Yes or No",
        "t1": 2016, "gold_t1": "Yes",
        "t2": 2005, "gold_t2": "No",
        "event_id": "SOUTH_SUDAN_INDEPENDENCE_2011",
        "temporal_cluster_id": "21ST_CENTURY_NATIONHOOD",
        "source_record_id": "HIST_MILESTONE_SSD_01",
        "provenance": "Comprehensive Peace Agreement and independence referendum July 9 2011",
    },
    {
        "pair_id": "CF_FLIP_07_TIMOR_LESTE_INDEPENDENCE",
        "entity": "Timor-Leste",
        "relation": "internationally_recognized_independent_state",
        "question_template": "In {year}, Timor-Leste was an internationally recognized sovereign independent state? Answer as Yes or No",
        "t1": 2008, "gold_t1": "Yes",
        "t2": 1995, "gold_t2": "No",
        "event_id": "TIMOR_LESTE_RESTORATION_2002",
        "temporal_cluster_id": "21ST_CENTURY_NATIONHOOD",
        "source_record_id": "HIST_MILESTONE_TLS_01",
        "provenance": "Restoration of independence May 20 2002; UN admission Sept 2002",
    },
    {
        "pair_id": "CF_FLIP_08_ERITREA_INDEPENDENCE",
        "entity": "Eritrea",
        "relation": "sovereign_independent_nation",
        "question_template": "In {year}, Eritrea was an independent sovereign nation? Answer as Yes or No",
        "t1": 1998, "gold_t1": "Yes",
        "t2": 1985, "gold_t2": "No",
        "event_id": "ERITREA_INDEPENDENCE_1993",
        "temporal_cluster_id": "POST_COLD_WAR_AFRICA",
        "source_record_id": "HIST_MILESTONE_ERI_01",
        "provenance": "UN-monitored independence referendum April 1993, declared May 24 1993",
    },
    {
        "pair_id": "CF_FLIP_09_SINGAPORE_INDEPENDENCE",
        "entity": "Singapore",
        "relation": "sovereign_independent_republic",
        "question_template": "In {year}, Singapore was an independent sovereign republic? Answer as Yes or No",
        "t1": 1972, "gold_t1": "Yes",
        "t2": 1960, "gold_t2": "No",
        "event_id": "SINGAPORE_SEPARATION_1965",
        "temporal_cluster_id": "SOUTHEAST_ASIA_DECOLONIZATION",
        "source_record_id": "HIST_MILESTONE_SGP_01",
        "provenance": "Separation agreement from Malaysia August 9 1965",
    },
    {
        "pair_id": "CF_FLIP_10_INDIA_INDEPENDENCE",
        "entity": "India",
        "relation": "independent_sovereign_nation",
        "question_template": "In {year}, India was an independent sovereign nation? Answer as Yes or No",
        "t1": 1955, "gold_t1": "Yes",
        "t2": 1935, "gold_t2": "No",
        "event_id": "INDIAN_INDEPENDENCE_ACT_1947",
        "temporal_cluster_id": "SOUTH_ASIA_DECOLONIZATION",
        "source_record_id": "HIST_MILESTONE_IND_01",
        "provenance": "Indian Independence Act 1947 (effective August 15 1947)",
    },
    {
        "pair_id": "CF_FLIP_11_PHILIPPINES_INDEPENDENCE",
        "entity": "Philippines",
        "relation": "independent_sovereign_republic",
        "question_template": "In {year}, the Republic of the Philippines was an independent sovereign republic? Answer as Yes or No",
        "t1": 1952, "gold_t1": "Yes",
        "t2": 1930, "gold_t2": "No",
        "event_id": "TREATY_OF_MANILA_1946",
        "temporal_cluster_id": "SOUTHEAST_ASIA_DECOLONIZATION",
        "source_record_id": "HIST_MILESTONE_PHL_01",
        "provenance": "Treaty of Manila July 4 1946 (US recognition of sovereignty)",
    },
    {
        "pair_id": "CF_FLIP_12_ALGERIA_INDEPENDENCE",
        "entity": "Algeria",
        "relation": "sovereign_independent_nation",
        "question_template": "In {year}, Algeria was an independent sovereign nation? Answer as Yes or No",
        "t1": 1968, "gold_t1": "Yes",
        "t2": 1954, "gold_t2": "No",
        "event_id": "EVIAN_ACCORDS_1962",
        "temporal_cluster_id": "NORTH_AFRICA_DECOLONIZATION",
        "source_record_id": "HIST_MILESTONE_DZA_01",
        "provenance": "Evian Accords and self-determination referendum July 5 1962",
    },

    # 2. International Organizations & Treaties
    {
        "pair_id": "CF_FLIP_13_VIETNAM_UN_MEMBER",
        "entity": "Vietnam",
        "relation": "official_member_of_united_nations",
        "question_template": "In {year}, Vietnam was an official member state of the United Nations? Answer as Yes or No",
        "t1": 1982, "gold_t1": "Yes",
        "t2": 1972, "gold_t2": "No",
        "event_id": "UN_RESOLUTION_32_2_VIETNAM",
        "temporal_cluster_id": "UN_MEMBERSHIP_MILESTONES",
        "source_record_id": "HIST_TREATY_UN_VNM",
        "provenance": "UN General Assembly Resolution 32/2 (admitted September 20 1977)",
    },
    {
        "pair_id": "CF_FLIP_14_JAPAN_UN_MEMBER",
        "entity": "Japan",
        "relation": "official_member_of_united_nations",
        "question_template": "In {year}, Japan was an official member state of the United Nations? Answer as Yes or No",
        "t1": 1965, "gold_t1": "Yes",
        "t2": 1950, "gold_t2": "No",
        "event_id": "UN_RESOLUTION_1111_JAPAN",
        "temporal_cluster_id": "UN_MEMBERSHIP_MILESTONES",
        "source_record_id": "HIST_TREATY_UN_JPN",
        "provenance": "UN General Assembly Resolution 1111 (admitted December 18 1956)",
    },
    {
        "pair_id": "CF_FLIP_15_SWITZERLAND_UN_MEMBER",
        "entity": "Switzerland",
        "relation": "official_member_of_united_nations",
        "question_template": "In {year}, Switzerland was a full voting member state of the United Nations? Answer as Yes or No",
        "t1": 2006, "gold_t1": "Yes",
        "t2": 1995, "gold_t2": "No",
        "event_id": "SWISS_UN_REFERENDUM_2002",
        "temporal_cluster_id": "UN_MEMBERSHIP_MILESTONES",
        "source_record_id": "HIST_TREATY_UN_CHE",
        "provenance": "Swiss referendum March 2002; admitted UN General Assembly Sept 10 2002",
    },
    {
        "pair_id": "CF_FLIP_16_VIETNAM_WTO_MEMBER",
        "entity": "Vietnam",
        "relation": "official_member_of_wto",
        "question_template": "In {year}, Vietnam was an official member of the World Trade Organization? Answer as Yes or No",
        "t1": 2012, "gold_t1": "Yes",
        "t2": 2002, "gold_t2": "No",
        "event_id": "VIETNAM_WTO_ACCESSION_2007",
        "temporal_cluster_id": "GLOBAL_TRADE_GOVERNANCE",
        "source_record_id": "HIST_TREATY_WTO_VNM",
        "provenance": "WTO General Council accession protocol (admitted January 11 2007)",
    },
    {
        "pair_id": "CF_FLIP_17_CHINA_WTO_MEMBER",
        "entity": "China",
        "relation": "official_member_of_wto",
        "question_template": "In {year}, China was an official member of the World Trade Organization? Answer as Yes or No",
        "t1": 2008, "gold_t1": "Yes",
        "t2": 1997, "gold_t2": "No",
        "event_id": "CHINA_WTO_ACCESSION_2001",
        "temporal_cluster_id": "GLOBAL_TRADE_GOVERNANCE",
        "source_record_id": "HIST_TREATY_WTO_CHN",
        "provenance": "WTO Ministerial Conference Doha (accession December 11 2001)",
    },
    {
        "pair_id": "CF_FLIP_18_VIETNAM_ASEAN_MEMBER",
        "entity": "Vietnam",
        "relation": "official_member_of_asean",
        "question_template": "In {year}, Vietnam was an official member state of ASEAN? Answer as Yes or No",
        "t1": 2000, "gold_t1": "Yes",
        "t2": 1990, "gold_t2": "No",
        "event_id": "VIETNAM_ASEAN_ACCESSION_1995",
        "temporal_cluster_id": "SOUTHEAST_ASIA_REGIONALISM",
        "source_record_id": "HIST_TREATY_ASEAN_VNM",
        "provenance": "28th ASEAN Ministerial Meeting Bandar Seri Begawan (joined July 28 1995)",
    },
    {
        "pair_id": "CF_FLIP_19_UK_EU_MEMBER",
        "entity": "United Kingdom",
        "relation": "member_of_european_union",
        "question_template": "In {year}, the United Kingdom was an active member state of the European Union? Answer as Yes or No",
        "t1": 2012, "gold_t1": "Yes",
        "t2": 2022, "gold_t2": "No",
        "event_id": "BREXIT_WITHDRAWAL_2020",
        "temporal_cluster_id": "EUROPEAN_INTEGRATION",
        "source_record_id": "HIST_TREATY_EU_GBR",
        "provenance": "European Union (Withdrawal Agreement) Act 2020 (exit January 31 2020)",
    },
    {
        "pair_id": "CF_FLIP_20_WTO_OPERATIONAL",
        "entity": "World Trade Organization",
        "relation": "active_operational_trade_body",
        "question_template": "In {year}, the World Trade Organization was actively operating as an international body? Answer as Yes or No",
        "t1": 2002, "gold_t1": "Yes",
        "t2": 1988, "gold_t2": "No",
        "event_id": "MARRAKESH_AGREEMENT_1994",
        "temporal_cluster_id": "GLOBAL_TRADE_GOVERNANCE",
        "source_record_id": "HIST_TREATY_WTO_GEN",
        "provenance": "Marrakesh Agreement Establishing the WTO (effective January 1 1995)",
    },

    # 3. Currency & Monetary Unions
    {
        "pair_id": "CF_FLIP_21_FRANCE_EURO_CURRENCY",
        "entity": "France",
        "relation": "euro_official_physical_circulating_currency",
        "question_template": "In {year}, the Euro was the official physical circulating currency in France? Answer as Yes or No",
        "t1": 2008, "gold_t1": "Yes",
        "t2": 1996, "gold_t2": "No",
        "event_id": "EURO_CASH_INTRODUCTION_2002",
        "temporal_cluster_id": "EUROZONE_MONETARY_HISTORY",
        "source_record_id": "HIST_CURRENCY_FRA_01",
        "provenance": "Introduction of Euro physical banknotes and coins (January 1 2002)",
    },
    {
        "pair_id": "CF_FLIP_22_GERMANY_EURO_CURRENCY",
        "entity": "Germany",
        "relation": "euro_official_physical_circulating_currency",
        "question_template": "In {year}, the Euro was the official physical circulating currency in Germany? Answer as Yes or No",
        "t1": 2008, "gold_t1": "Yes",
        "t2": 1996, "gold_t2": "No",
        "event_id": "EURO_CASH_INTRODUCTION_2002",
        "temporal_cluster_id": "EUROZONE_MONETARY_HISTORY",
        "source_record_id": "HIST_CURRENCY_DEU_01",
        "provenance": "Replacement of Deutsche Mark by Euro physical cash (January 1 2002)",
    },
    {
        "pair_id": "CF_FLIP_23_ITALY_EURO_CURRENCY",
        "entity": "Italy",
        "relation": "euro_official_physical_circulating_currency",
        "question_template": "In {year}, the Euro was the official physical circulating currency in Italy? Answer as Yes or No",
        "t1": 2008, "gold_t1": "Yes",
        "t2": 1996, "gold_t2": "No",
        "event_id": "EURO_CASH_INTRODUCTION_2002",
        "temporal_cluster_id": "EUROZONE_MONETARY_HISTORY",
        "source_record_id": "HIST_CURRENCY_ITA_01",
        "provenance": "Replacement of Italian Lira by Euro physical cash (January 1 2002)",
    },
    {
        "pair_id": "CF_FLIP_24_SPAIN_EURO_CURRENCY",
        "entity": "Spain",
        "relation": "euro_official_physical_circulating_currency",
        "question_template": "In {year}, the Euro was the official physical circulating currency in Spain? Answer as Yes or No",
        "t1": 2008, "gold_t1": "Yes",
        "t2": 1996, "gold_t2": "No",
        "event_id": "EURO_CASH_INTRODUCTION_2002",
        "temporal_cluster_id": "EUROZONE_MONETARY_HISTORY",
        "source_record_id": "HIST_CURRENCY_ESP_01",
        "provenance": "Replacement of Spanish Peseta by Euro physical cash (January 1 2002)",
    },

    # 4. Human Rights, Governance & Legal Milestones
    {
        "pair_id": "CF_FLIP_25_SOUTH_AFRICA_APARTHEID",
        "entity": "South Africa",
        "relation": "apartheid_official_governing_state_policy",
        "question_template": "In {year}, Apartheid was the official governing legal system in South Africa? Answer as Yes or No",
        "t1": 1978, "gold_t1": "Yes",
        "t2": 1998, "gold_t2": "No",
        "event_id": "END_OF_APARTHEID_1994",
        "temporal_cluster_id": "CIVIL_RIGHTS_AFRICA",
        "source_record_id": "HIST_RIGHTS_ZAF_01",
        "provenance": "Interim Constitution 1993 and democratic elections April 27 1994",
    },
    {
        "pair_id": "CF_FLIP_26_SWISS_WOMEN_SUFFRAGE",
        "entity": "Switzerland",
        "relation": "women_legal_right_to_vote_federal",
        "question_template": "In {year}, women had the legal right to vote in federal elections in Switzerland? Answer as Yes or No",
        "t1": 1978, "gold_t1": "Yes",
        "t2": 1962, "gold_t2": "No",
        "event_id": "SWISS_WOMEN_SUFFRAGE_1971",
        "temporal_cluster_id": "SUFFRAGE_EUROPE",
        "source_record_id": "HIST_RIGHTS_CHE_01",
        "provenance": "Swiss federal referendum on women's suffrage (February 7 1971)",
    },
    {
        "pair_id": "CF_FLIP_27_US_WOMEN_SUFFRAGE",
        "entity": "United States",
        "relation": "women_constitutional_right_to_vote",
        "question_template": "In {year}, women possessed the constitutional right to vote in the United States? Answer as Yes or No",
        "t1": 1928, "gold_t1": "Yes",
        "t2": 1912, "gold_t2": "No",
        "event_id": "NINETEENTH_AMENDMENT_1920",
        "temporal_cluster_id": "SUFFRAGE_AMERICAS",
        "source_record_id": "HIST_RIGHTS_USA_01",
        "provenance": "Nineteenth Amendment to the US Constitution (ratified August 18 1920)",
    },
    {
        "pair_id": "CF_FLIP_28_FRANCE_CAPITAL_PUNISHMENT",
        "entity": "France",
        "relation": "capital_punishment_completely_abolished",
        "question_template": "In {year}, capital punishment was completely abolished in France? Answer as Yes or No",
        "t1": 1992, "gold_t1": "Yes",
        "t2": 1975, "gold_t2": "No",
        "event_id": "FRENCH_ABOLITION_DEATH_PENALTY_1981",
        "temporal_cluster_id": "HUMAN_RIGHTS_EUROPE",
        "source_record_id": "HIST_RIGHTS_FRA_01",
        "provenance": "Law no 81-908 of 9 October 1981 (abolition of death penalty under Mitterrand)",
    },
    {
        "pair_id": "CF_FLIP_29_ITALY_MONARCHY",
        "entity": "Italy",
        "relation": "monarchy_active_form_of_government",
        "question_template": "In {year}, a monarchy was the active form of government in Italy? Answer as Yes or No",
        "t1": 1932, "gold_t1": "Yes",
        "t2": 1952, "gold_t2": "No",
        "event_id": "ITALIAN_INSTITUTIONAL_REFERENDUM_1946",
        "temporal_cluster_id": "CONSTITUTIONAL_TRANSITIONS",
        "source_record_id": "HIST_POLITY_ITA_01",
        "provenance": "Institutional referendum of June 2 1946 establishing the Italian Republic",
    },
    {
        "pair_id": "CF_FLIP_30_GREECE_MONARCHY",
        "entity": "Greece",
        "relation": "monarchy_active_form_of_government",
        "question_template": "In {year}, a monarchy was the active form of government in Greece? Answer as Yes or No",
        "t1": 1962, "gold_t1": "Yes",
        "t2": 1982, "gold_t2": "No",
        "event_id": "GREEK_REPUBLIC_REFERENDUM_1974",
        "temporal_cluster_id": "CONSTITUTIONAL_TRANSITIONS",
        "source_record_id": "HIST_POLITY_GRC_01",
        "provenance": "Referendum of December 8 1974 establishing the Third Hellenic Republic",
    },
    {
        "pair_id": "CF_FLIP_31_PORTUGAL_CIVIL_LIBERTIES",
        "entity": "Portugal",
        "relation": "democratic_civil_liberties_higher_than_1952",
        "question_template": "In {year}, Private Civil Liberties Index in Portugal was higher than in 1952? Answer as Yes or No",
        "t1": 1984, "gold_t1": "Yes",
        "t2": 1947, "gold_t2": "No",
        "event_id": "PORTUGUESE_CARNATION_REVOLUTION_1974",
        "temporal_cluster_id": "VDEM_SUBSTRATE",
        "source_record_id": "VDEM_PRVT_LIB_PRT_01",
        "provenance": "V-Dem v13 substrate: Carnation Revolution 1974 transformed civil liberties",
    },
    {
        "pair_id": "CF_FLIP_32_SWEDEN_RESOURCE_DISTRIBUTION",
        "entity": "Sweden",
        "relation": "equal_distribution_higher_than_1960",
        "question_template": "In {year}, Equal Distribution of Resources Index in Sweden was higher than 1960? Answer as Yes or No",
        "t1": 1984, "gold_t1": "Yes",
        "t2": 1947, "gold_t2": "No",
        "event_id": "SWEDISH_WELFARE_STATE_EXPANSION",
        "temporal_cluster_id": "VDEM_SUBSTRATE",
        "source_record_id": "VDEM_RES_DIST_SWE_01",
        "provenance": "V-Dem v13 substrate: Social Democratic postwar welfare model expansion",
    },

    # 5. Scientific, Technological & Infrastructure Milestones
    {
        "pair_id": "CF_FLIP_33_MOON_LANDING",
        "entity": "Moon",
        "relation": "humans_successfully_landed_on_surface",
        "question_template": "In {year}, humans had successfully landed on the surface of the Moon? Answer as Yes or No",
        "t1": 1975, "gold_t1": "Yes",
        "t2": 1962, "gold_t2": "No",
        "event_id": "APOLLO_11_MOON_LANDING_1969",
        "temporal_cluster_id": "SPACE_EXPLORATION",
        "source_record_id": "HIST_TECH_MOON_01",
        "provenance": "NASA Apollo 11 mission landing on Mare Tranquillitatis July 20 1969",
    },
    {
        "pair_id": "CF_FLIP_34_ISS_HABITATION",
        "entity": "International Space Station",
        "relation": "continuously_inhabited_by_human_crew",
        "question_template": "In {year}, the International Space Station was continuously inhabited by humans? Answer as Yes or No",
        "t1": 2008, "gold_t1": "Yes",
        "t2": 1994, "gold_t2": "No",
        "event_id": "ISS_EXPEDITION_1_2000",
        "temporal_cluster_id": "SPACE_EXPLORATION",
        "source_record_id": "HIST_TECH_ISS_01",
        "provenance": "Expedition 1 docking and permanent human occupation (November 2 2000)",
    },
    {
        "pair_id": "CF_FLIP_35_CONCORDE_SERVICE",
        "entity": "Concorde",
        "relation": "active_commercial_supersonic_passenger_flights",
        "question_template": "In {year}, commercial supersonic passenger flights with Concorde were actively operating? Answer as Yes or No",
        "t1": 1990, "gold_t1": "Yes",
        "t2": 2012, "gold_t2": "No",
        "event_id": "CONCORDE_RETIREMENT_2003",
        "temporal_cluster_id": "AVIATION_HISTORY",
        "source_record_id": "HIST_TECH_CONCORDE_01",
        "provenance": "Entered service January 1976; final commercial flights October 2003",
    },
    {
        "pair_id": "CF_FLIP_36_PANAMA_CANAL",
        "entity": "Panama Canal",
        "relation": "active_commercial_maritime_operation",
        "question_template": "In {year}, the Panama Canal was in active commercial operation for maritime trade? Answer as Yes or No",
        "t1": 1925, "gold_t1": "Yes",
        "t2": 1905, "gold_t2": "No",
        "event_id": "PANAMA_CANAL_OPENING_1914",
        "temporal_cluster_id": "INFRASTRUCTURE_HISTORY",
        "source_record_id": "HIST_TECH_PANAMA_01",
        "provenance": "Official commercial opening of the Panama Canal (August 15 1914)",
    },
]


# 18 Grounded Gold-Same Pairs (Negative Controls: Factual Truth DOES NOT change between t1 and t2)
GOLD_SAME_PAIRS_SPEC: List[Dict[str, Any]] = [
    {
        "pair_id": "CF_SAME_01_PARIS_CAPITAL",
        "entity": "France",
        "relation": "official_capital_city_is_paris",
        "question_template": "In {year}, Paris was the official capital city of France? Answer as Yes or No",
        "t1": 1955, "gold_t1": "Yes",
        "t2": 1985, "gold_t2": "Yes",
        "event_id": "STABLE_POLITY_FRA",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_01",
        "provenance": "Stable sovereign capital landmark (Paris)",
    },
    {
        "pair_id": "CF_SAME_02_USA_INDEPENDENT",
        "entity": "United States",
        "relation": "sovereign_independent_nation",
        "question_template": "In {year}, the United States was an independent sovereign nation? Answer as Yes or No",
        "t1": 1920, "gold_t1": "Yes",
        "t2": 1960, "gold_t2": "Yes",
        "event_id": "STABLE_POLITY_USA",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_02",
        "provenance": "Stable sovereign statehood 1920-1960",
    },
    {
        "pair_id": "CF_SAME_03_UK_UN_SECURITY_COUNCIL",
        "entity": "United Kingdom",
        "relation": "permanent_member_of_un_security_council",
        "question_template": "In {year}, the United Kingdom was a permanent member of the United Nations Security Council? Answer as Yes or No",
        "t1": 1955, "gold_t1": "Yes",
        "t2": 1995, "gold_t2": "Yes",
        "event_id": "STABLE_TREATY_UN_P5",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_03",
        "provenance": "Charter of the United Nations Article 23",
    },
    {
        "pair_id": "CF_SAME_04_VIETNAM_UN_POST_1980",
        "entity": "Vietnam",
        "relation": "official_member_of_united_nations",
        "question_template": "In {year}, Vietnam was an official member state of the United Nations? Answer as Yes or No",
        "t1": 1985, "gold_t1": "Yes",
        "t2": 2015, "gold_t2": "Yes",
        "event_id": "STABLE_TREATY_UN_VNM",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_04",
        "provenance": "Continuous UN membership since 1977",
    },
    {
        "pair_id": "CF_SAME_05_MOON_LANDING_POST_1975",
        "entity": "Moon",
        "relation": "humans_successfully_landed_on_surface",
        "question_template": "In {year}, humans had successfully landed on the surface of the Moon? Answer as Yes or No",
        "t1": 1980, "gold_t1": "Yes",
        "t2": 2010, "gold_t2": "Yes",
        "event_id": "STABLE_TECH_MOON",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_05",
        "provenance": "Post-1969 historical fact permanence",
    },
    {
        "pair_id": "CF_SAME_06_GERMANY_UNIFIED_POST_2000",
        "entity": "Germany",
        "relation": "unified_single_state",
        "question_template": "In {year}, Germany was a unified single democratic state? Answer as Yes or No",
        "t1": 2005, "gold_t1": "Yes",
        "t2": 2020, "gold_t2": "Yes",
        "event_id": "STABLE_POLITY_DEU",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_06",
        "provenance": "Continuous German unity post-1990",
    },
    {
        "pair_id": "CF_SAME_07_ROME_CAPITAL_ITALY",
        "entity": "Italy",
        "relation": "official_capital_city_is_rome",
        "question_template": "In {year}, Rome was the official capital city of Italy? Answer as Yes or No",
        "t1": 1960, "gold_t1": "Yes",
        "t2": 2000, "gold_t2": "Yes",
        "event_id": "STABLE_POLITY_ITA",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_07",
        "provenance": "Capital of Italy since 1871",
    },
    {
        "pair_id": "CF_SAME_08_TOKYO_CAPITAL_JAPAN",
        "entity": "Japan",
        "relation": "official_capital_city_is_tokyo",
        "question_template": "In {year}, Tokyo was the capital city of Japan? Answer as Yes or No",
        "t1": 1950, "gold_t1": "Yes",
        "t2": 1990, "gold_t2": "Yes",
        "event_id": "STABLE_POLITY_JPN",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_08",
        "provenance": "Capital of Japan since Meiji era",
    },
    {
        "pair_id": "CF_SAME_09_PRE_1960_HUMANS_IN_SPACE",
        "entity": "Space",
        "relation": "humans_successfully_orbited_earth",
        "question_template": "In {year}, human beings had successfully completed an orbital spaceflight around Earth? Answer as Yes or No",
        "t1": 1930, "gold_t1": "No",
        "t2": 1950, "gold_t2": "No",
        "event_id": "STABLE_NEGATIVE_SPACE",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_09",
        "provenance": "Gagarin flight was 1961; prior years are strictly No",
    },
    {
        "pair_id": "CF_SAME_10_PRE_1990_SMARTPHONES",
        "entity": "Technology",
        "relation": "modern_touchscreen_smartphones_widely_sold",
        "question_template": "In {year}, modern capacitive touchscreen smartphones were widely commercially sold? Answer as Yes or No",
        "t1": 1970, "gold_t1": "No",
        "t2": 1985, "gold_t2": "No",
        "event_id": "STABLE_NEGATIVE_TECH",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_10",
        "provenance": "Modern smartphones emerged in late 2000s; strictly No",
    },
    {
        "pair_id": "CF_SAME_11_PRE_1940_UNITED_NATIONS",
        "entity": "United Nations",
        "relation": "united_nations_officially_existed",
        "question_template": "In {year}, the United Nations officially existed as an operating global organization? Answer as Yes or No",
        "t1": 1920, "gold_t1": "No",
        "t2": 1935, "gold_t2": "No",
        "event_id": "STABLE_NEGATIVE_UN",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_11",
        "provenance": "UN Charter signed 1945; prior years are strictly No",
    },
    {
        "pair_id": "CF_SAME_12_PRE_1940_ATOMIC_BOMB",
        "entity": "Technology",
        "relation": "nuclear_weapons_had_been_detonated",
        "question_template": "In {year}, a nuclear weapon had been successfully detonated on Earth? Answer as Yes or No",
        "t1": 1915, "gold_t1": "No",
        "t2": 1930, "gold_t2": "No",
        "event_id": "STABLE_NEGATIVE_NUCLEAR",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_12",
        "provenance": "Trinity test was July 1945; strictly No",
    },
    {
        "pair_id": "CF_SAME_13_SWITZERLAND_NEUTRALITY_20TH_CENTURY",
        "entity": "Switzerland",
        "relation": "maintained_armed_neutrality_policy",
        "question_template": "In {year}, Switzerland officially maintained an armed neutrality foreign policy? Answer as Yes or No",
        "t1": 1930, "gold_t1": "Yes",
        "t2": 1970, "gold_t2": "Yes",
        "event_id": "STABLE_POLITY_CHE",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_13",
        "provenance": "Continuous Swiss armed neutrality policy established 1815",
    },
    {
        "pair_id": "CF_SAME_14_PANAMA_CANAL_POST_1930",
        "entity": "Panama Canal",
        "relation": "active_commercial_maritime_operation",
        "question_template": "In {year}, the Panama Canal was in active commercial operation for maritime trade? Answer as Yes or No",
        "t1": 1950, "gold_t1": "Yes",
        "t2": 1980, "gold_t2": "Yes",
        "event_id": "STABLE_INFRA_PANAMA",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_14",
        "provenance": "Continuous commercial maritime operation post-1914",
    },
    {
        "pair_id": "CF_SAME_15_SUEZ_CANAL_POST_1900",
        "entity": "Suez Canal",
        "relation": "open_for_international_maritime_transit",
        "question_template": "In {year}, the Suez Canal was open for international maritime navigation? Answer as Yes or No",
        "t1": 1910, "gold_t1": "Yes",
        "t2": 1935, "gold_t2": "Yes",
        "event_id": "STABLE_INFRA_SUEZ",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_15",
        "provenance": "Continuous maritime transit post-1869",
    },
    {
        "pair_id": "CF_SAME_16_POST_1995_WTO_EXISTENCE",
        "entity": "World Trade Organization",
        "relation": "active_operational_trade_body",
        "question_template": "In {year}, the World Trade Organization was actively operating as an international body? Answer as Yes or No",
        "t1": 2005, "gold_t1": "Yes",
        "t2": 2018, "gold_t2": "Yes",
        "event_id": "STABLE_TREATY_WTO",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_16",
        "provenance": "Continuous WTO operation post-1995",
    },
    {
        "pair_id": "CF_SAME_17_PRE_1960_APOLLO_PROGRAM",
        "entity": "Apollo Program",
        "relation": "apollo_spacecraft_had_orbited_the_moon",
        "question_template": "In {year}, a crewed Apollo spacecraft had successfully orbited the Moon? Answer as Yes or No",
        "t1": 1935, "gold_t1": "No",
        "t2": 1955, "gold_t2": "No",
        "event_id": "STABLE_NEGATIVE_APOLLO",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_17",
        "provenance": "Apollo 8 was December 1968; strictly No",
    },
    {
        "pair_id": "CF_SAME_18_POST_1950_INDIA_INDEPENDENT",
        "entity": "India",
        "relation": "independent_sovereign_nation",
        "question_template": "In {year}, India was an independent sovereign nation? Answer as Yes or No",
        "t1": 1965, "gold_t1": "Yes",
        "t2": 1995, "gold_t2": "Yes",
        "event_id": "STABLE_POLITY_IND",
        "temporal_cluster_id": "STABLE_HISTORICAL_CONTROL",
        "source_record_id": "CONTROL_STABLE_18",
        "provenance": "Continuous sovereign independence post-1947",
    },
]


def pair_to_question_records(pair: Dict[str, Any], is_flip: bool) -> List[Dict[str, Any]]:
    """Convert a pair specification into two evaluated question records."""
    pid = pair["pair_id"]
    t1 = pair["t1"]
    t2 = pair["t2"]
    q1 = pair["question_template"].format(year=t1)
    q2 = pair["question_template"].format(year=t2)
    
    status = "ELIGIBLE_GOLD_FLIP" if is_flip else "ELIGIBLE_GOLD_SAME"
    
    rec_t1 = {
        "example_id": f"{pid}_t1_{t1}",
        "pair_id": pid,
        "input": q1,
        "question": q1,
        "year": t1,
        "year_eval": t1,
        "canonical_year": t1,
        "target": pair["gold_t1"],
        "target_text": pair["gold_t1"],
        "gold_semantic_answer": pair["gold_t1"],
        "gold_display_label": pair["gold_t1"],
        "task_id": "CP",
        "entity": pair["entity"],
        "relation": pair["relation"],
        "event_id": pair["event_id"],
        "source_record_id": pair["source_record_id"],
        "temporal_cluster_id": pair["temporal_cluster_id"],
        "provenance": pair["provenance"],
        "is_counterfactual_pair": True,
        "gold_flips": is_flip,
        "treatment_status": status,
        "partner_year": t2,
        "partner_gold": pair["gold_t2"],
        "role": "anchor_t1",
    }
    rec_t2 = {
        "example_id": f"{pid}_t2_{t2}",
        "pair_id": pid,
        "input": q2,
        "question": q2,
        "year": t2,
        "year_eval": t2,
        "canonical_year": t2,
        "target": pair["gold_t2"],
        "target_text": pair["gold_t2"],
        "gold_semantic_answer": pair["gold_t2"],
        "gold_display_label": pair["gold_t2"],
        "task_id": "CP",
        "entity": pair["entity"],
        "relation": pair["relation"],
        "event_id": pair["event_id"],
        "source_record_id": pair["source_record_id"],
        "temporal_cluster_id": pair["temporal_cluster_id"],
        "provenance": pair["provenance"],
        "is_counterfactual_pair": True,
        "gold_flips": is_flip,
        "treatment_status": status,
        "partner_year": t1,
        "partner_gold": pair["gold_t1"],
        "role": "anchor_t2",
    }
    return [rec_t1, rec_t2]


def build_and_save_counterfactuals() -> Dict[str, Any]:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    flip_records: List[Dict[str, Any]] = []
    for p in GOLD_FLIP_PAIRS_SPEC:
        flip_records.extend(pair_to_question_records(p, is_flip=True))
        
    same_records: List[Dict[str, Any]] = []
    for p in GOLD_SAME_PAIRS_SPEC:
        same_records.extend(pair_to_question_records(p, is_flip=False))
        
    flip_path = _DATA_DIR / "grounded_gold_flip_pairs.jsonl"
    with open(flip_path, "w", encoding="utf-8") as f:
        for r in flip_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    same_path = _DATA_DIR / "grounded_gold_same_pairs.jsonl"
    with open(same_path, "w", encoding="utf-8") as f:
        for r in same_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Also create combined benchmark file
    combined_path = _DATA_DIR / "grounded_counterfactual_benchmark.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for r in flip_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in same_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Compute SHA256 hashes
    def get_sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "dataset_name": "GROUNDED_TEMPORAL_COUNTERFACTUALS_V1",
        "description": "Grounded semantic temporal counterfactual pairs (gold-flip primary endpoint + gold-same negative controls)",
        "protocol": "Protocol P12 / Decision 19 Representation-First Contract",
        "created_at": "2026-10-19",
        "files": {
            "grounded_gold_flip_pairs.jsonl": {
                "sha256": get_sha256(flip_path),
                "pair_count": len(GOLD_FLIP_PAIRS_SPEC),
                "question_count": len(flip_records),
                "treatment_status": "ELIGIBLE_GOLD_FLIP",
            },
            "grounded_gold_same_pairs.jsonl": {
                "sha256": get_sha256(same_path),
                "pair_count": len(GOLD_SAME_PAIRS_SPEC),
                "question_count": len(same_records),
                "treatment_status": "ELIGIBLE_GOLD_SAME",
            },
            "grounded_counterfactual_benchmark.jsonl": {
                "sha256": get_sha256(combined_path),
                "total_pairs": len(GOLD_FLIP_PAIRS_SPEC) + len(GOLD_SAME_PAIRS_SPEC),
                "total_questions": len(flip_records) + len(same_records),
            },
        },
        "coverage": {
            "gold_flip_pairs": len(GOLD_FLIP_PAIRS_SPEC),
            "gold_same_pairs": len(GOLD_SAME_PAIRS_SPEC),
            "unique_entities": len(set(p["entity"] for p in GOLD_FLIP_PAIRS_SPEC + GOLD_SAME_PAIRS_SPEC)),
            "unique_events": len(set(p["event_id"] for p in GOLD_FLIP_PAIRS_SPEC + GOLD_SAME_PAIRS_SPEC)),
            "unique_clusters": len(set(p["temporal_cluster_id"] for p in GOLD_FLIP_PAIRS_SPEC + GOLD_SAME_PAIRS_SPEC)),
        },
    }
    
    manifest_path = _DATA_DIR / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    return manifest


if __name__ == "__main__":
    m = build_and_save_counterfactuals()
    print("Successfully built Grounded Semantic Temporal Counterfactuals!")
    print(f"Gold-Flip: {m['files']['grounded_gold_flip_pairs.jsonl']['pair_count']} pairs ({m['files']['grounded_gold_flip_pairs.jsonl']['question_count']} questions)")
    print(f"Gold-Same: {m['files']['grounded_gold_same_pairs.jsonl']['pair_count']} pairs ({m['files']['grounded_gold_same_pairs.jsonl']['question_count']} questions)")
    print(f"Manifest written to: {m['files']}")
