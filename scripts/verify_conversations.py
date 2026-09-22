import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.knowledge.mock import MockCompanyKnowledgeService

from app.services.conversation import ConversationService
from app.schemas.conversation import ConversationTurnRequest


async def run_conversations():
    service = ConversationService(knowledge_service=MockCompanyKnowledgeService())

    # 1. SALES CONVERSATION
    print("=" * 70)
    print("                    1. SALES CONVERSATION FLOW")
    print("=" * 70)
    sales_messages = [
        "hello",
        "what do you sell?",
        "what is airmega?",
        "I want one for my bedroom",
        "200 sqft",
        "yes",
        "tell me more",
        "how much?",
        "what offers are going on?",
        "what about warranty?",
        "I'm done",
    ]
    s_session = "live_sales_demo"
    for msg in sales_messages:
        print(f'Customer: "{msg}"')
        res = await service.process_message(ConversationTurnRequest(session_id=s_session, message=msg, department="SALES"))
        clean_msg = res.message.encode("ascii", "ignore").decode("ascii")
        print(f'AI: "{clean_msg}"')

        # Attribution layer
        if res.human_escalation_required:
            layer = "Deterministic System Event (Human Escalation Desk)"
        elif res.manager_intervention:
            layer = "Manager Supervisor (Safe Guidance / Transfer)"
        elif res.citations:
            layer = f"Specialist LLM Brain + Grounded Knowledge ({res.citations[0]})"
        else:
            layer = "Specialist LLM Brain (Conversation Context)"

        print(f"  [DEBUG: Agent={res.agent_role.value.upper()} | Dept={res.department} | LLM_USED={res.llm_used} | Model={res.llm_model} | Layer={layer}]\n")

    # 2. SUPPORT CONVERSATION
    print("=" * 70)
    print("                    2. SUPPORT CONVERSATION FLOW")
    print("=" * 70)
    support_messages = [
        "hello",
        "I have an issue with Airmega 250.",
        "yes",
        "silent",
        "removed the panel now?",
        "no unit does not restart",
        "I need a human",
    ]
    sup_session = "live_support_demo"
    for msg in support_messages:
        print(f'Customer: "{msg}"')
        res = await service.process_message(ConversationTurnRequest(session_id=sup_session, message=msg, department="CUSTOMER_SERVICE"))
        clean_msg = res.message.encode("ascii", "ignore").decode("ascii")
        print(f'AI: "{clean_msg}"')

        if res.human_escalation_required:
            layer = "Deterministic System Event (Human Escalation Desk)"
        elif res.manager_intervention:
            layer = "Manager Supervisor (Safe Guidance / Transfer)"
        elif res.citations:
            layer = f"Specialist LLM Brain + Grounded Knowledge ({res.citations[0]})"
        else:
            layer = "Specialist LLM Brain (Conversation Context)"

        stage_val = res.stage.value if hasattr(res.stage, "value") else str(res.stage)
        print(f"  [DEBUG: Agent={res.agent_role.value.upper()} | Dept={res.department} | Stage={stage_val} | Intent={res.intent.value} | LLM_USED={res.llm_used} | Model={res.llm_model} | Layer={layer}]\n")

    # 3. CONTEXT SWITCHING CONVERSATION
    print("=" * 70)
    print("                3. CONTEXT SWITCHING CONVERSATION FLOW")
    print("=" * 70)
    switch_messages = [
        "I'm looking for a purifier for my bedroom.",
        "200 sqft.",
        "How much?",
        "And warranty?",
        "Actually, what about a bigger room?",
    ]
    switch_session = "live_switch_demo"
    for msg in switch_messages:
        print(f'Customer: "{msg}"')
        res = await service.process_message(ConversationTurnRequest(session_id=switch_session, message=msg, department="SALES"))
        clean_msg = res.message.encode("ascii", "ignore").decode("ascii")
        print(f'AI: "{clean_msg}"')

        if res.human_escalation_required:
            layer = "Deterministic System Event (Human Escalation Desk)"
        elif res.manager_intervention:
            layer = "Manager Supervisor (Safe Guidance / Transfer)"
        elif res.citations:
            layer = f"Specialist LLM Brain + Grounded Knowledge ({res.citations[0]})"
        else:
            layer = "Specialist LLM Brain (Conversation Context)"

        print(f"  [DEBUG: Agent={res.agent_role.value.upper()} | Dept={res.department} | LLM_USED={res.llm_used} | Model={res.llm_model} | Layer={layer}]\n")

    # 4. REGRESSION TESTS (TESTS A THROUGH F)
    print("=" * 70)
    print("                4. REGRESSION VERIFICATION (TESTS A - F)")
    print("=" * 70)

    # Test A: Comparison
    print("\n--- TEST A: Comparison ---")
    s_a = "reg_test_a"
    res_a = await service.process_message(ConversationTurnRequest(session_id=s_a, message="what is airmega 150 vs 250"))
    print(f'Customer: "what is airmega 150 vs 250"')
    print(f'AI: "{res_a.message.encode("ascii", "ignore").decode("ascii")}"')
    print(f'  [DEBUG: Knowledge_Retrieved={res_a.knowledge_retrieved} | Citations={res_a.citations}]')

    # Test B: Active Product Pricing
    print("\n--- TEST B: Active Product Pricing (No 150 Leakage) ---")
    s_b = "reg_test_b"
    await service.process_message(ConversationTurnRequest(session_id=s_b, message="I want the Airmega 250"))
    res_b = await service.process_message(ConversationTurnRequest(session_id=s_b, message="how much?"))
    print(f'Customer: "I want the Airmega 250" -> "how much?"')
    print(f'AI: "{res_b.message.encode("ascii", "ignore").decode("ascii")}"')
    print(f'  [DEBUG: Knowledge_Retrieved={res_b.knowledge_retrieved} | Active_Product=250]')

    # Test C: Follow-up Pricing after Offers
    print("\n--- TEST C: Follow-up Pricing after Offers ---")
    s_c = "reg_test_c"
    await service.process_message(ConversationTurnRequest(session_id=s_c, message="what about offers on Airmega 250?"))
    res_c = await service.process_message(ConversationTurnRequest(session_id=s_c, message="cost?"))
    print(f'Customer: "what about offers on Airmega 250?" -> "cost?"')
    print(f'AI: "{res_c.message.encode("ascii", "ignore").decode("ascii")}"')
    print(f'  [DEBUG: Knowledge_Retrieved={res_c.knowledge_retrieved} | Active_Product=250]')

    # Test D: Broader Catalog
    print("\n--- TEST D: Broader Catalog ---")
    s_d = "reg_test_d"
    res_d = await service.process_message(ConversationTurnRequest(session_id=s_d, message="what else does Coway sell apart from Airmega?"))
    print(f'Customer: "what else does Coway sell apart from Airmega?"')
    print(f'AI: "{res_d.message.encode("ascii", "ignore").decode("ascii")}"')
    print(f'  [DEBUG: Knowledge_Retrieved={res_d.knowledge_retrieved} | Citations={res_d.citations}]')

    # Test E: Senior Executive
    print("\n--- TEST E: Senior Executive Escalation ---")
    s_e = "reg_test_e"
    res_e = await service.process_message(ConversationTurnRequest(session_id=s_e, message="get me a seniorexec"))
    print(f'Customer: "get me a seniorexec"')
    print(f'AI: "{res_e.message.encode("ascii", "ignore").decode("ascii")}"')
    print(f'  [DEBUG: Escalation_Required={res_e.human_escalation_required} | Status={res_e.resolution_status}]')


if __name__ == "__main__":
    asyncio.run(run_conversations())
