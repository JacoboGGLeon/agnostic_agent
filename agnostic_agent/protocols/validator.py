from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from agnostic_agent.protocols.kap import KnowledgeAdapterContract
from agnostic_agent.protocols.scp import CompositionPlan
from agnostic_agent.protocols.srp import SkillRuntimeResponse
from agnostic_agent.protocols.tcp import ToolContract
from agnostic_agent.protocols.tep import TEPBundle


def validate_scp_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        CompositionPlan(**plan)
        return True, []
    except ValidationError as e:
        return False, [str(err["msg"]) for err in e.errors()]


def validate_srp_response(response: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        SkillRuntimeResponse(**response)
        return True, []
    except ValidationError as e:
        return False, [str(err["msg"]) for err in e.errors()]


def validate_tcp_contract(contract: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        ToolContract(**contract)
        return True, []
    except ValidationError as e:
        return False, [str(err["msg"]) for err in e.errors()]


def validate_kap_contract(contract: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        KnowledgeAdapterContract(**contract)
        return True, []
    except ValidationError as e:
        return False, [str(err["msg"]) for err in e.errors()]


def validate_tep_bundle(bundle: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        TEPBundle(**bundle)
        return True, []
    except ValidationError as e:
        return False, [str(err["msg"]) for err in e.errors()]
