from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from customer_support_agent import CustomerSupportAgent, RouterDecision


def make_agent(temp_db_path):
    return CustomerSupportAgent(db_path=temp_db_path)


def base_state(**overrides):
    state = {
        "messages": [],
        "customer_id": None,
        "customer_name": None,
        "customer_tier": "standard",
        "active_agent": "supervisor",
        "resolved": False,
        "requires_escalation": False,
        "is_human_takeover": False,
        "ticket_id": None,
        "current_step": "start",
        "customer_sentiment": "neutral",
        "total_tokens": 0,
    }
    state.update(overrides)
    return state


def test_identify_node_recognizes_known_customer_email(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(messages=[HumanMessage(content="I'm alice@example.com, need help")])

    result = agent._identify_node(state)

    assert result["customer_id"] == "C1"
    assert result["customer_tier"] == "premium"


def test_identify_node_leaves_state_unchanged_for_unknown_email(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(messages=[HumanMessage(content="I'm nobody@nowhere.com")])

    result = agent._identify_node(state)

    assert result.get("customer_id") is None


def test_order_agent_reports_latest_order_for_known_customer(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(customer_id="C1", messages=[HumanMessage(content="Where's my order?")])

    result = agent._order_agent_node(state)

    assert "ORD-123" in result["messages"][0].content or "ORD-456" in result["messages"][0].content


def test_order_agent_asks_for_email_when_customer_unknown(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(messages=[HumanMessage(content="Where's my order?")])

    result = agent._order_agent_node(state)

    assert "email" in result["messages"][0].content.lower()


def test_tech_agent_matches_knowledge_base_entry(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(messages=[HumanMessage(content="I forgot my password")])

    result = agent._tech_agent_node(state)

    assert "Forgot Password" in result["messages"][0].content


def test_billing_agent_escalates_on_dispute_keywords(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(messages=[HumanMessage(content="I want to dispute this charge")])

    result = agent._billing_agent_node(state)

    assert result["active_agent"] == "escalate"


def test_billing_agent_handles_routine_query_without_escalating(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(messages=[HumanMessage(content="Is my account in good standing?")])

    result = agent._billing_agent_node(state)

    assert result.get("active_agent") != "escalate"


def test_escalation_node_creates_ticket_and_locks_human_takeover(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(
        customer_id="C1",
        messages=[HumanMessage(content="I need a refund, this is unacceptable")],
    )

    result = agent._escalation_node(state)

    assert result["is_human_takeover"] is True
    assert "Ticket #" in result["messages"][0].content


def test_supervisor_routes_based_on_router_decision(temp_db_path):
    agent = make_agent(temp_db_path)
    agent.router_chain = MagicMock()
    agent.router_chain.invoke.return_value = RouterDecision(
        next_agent="tech_specialist", reasoning="user mentioned a login issue"
    )
    state = base_state(messages=[HumanMessage(content="I can't log in")])

    result = agent._supervisor_node(state)

    assert result["active_agent"] == "tech_specialist"


def test_supervisor_ends_turn_after_ai_message(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(messages=[HumanMessage(content="hi"), AIMessage(content="how can I help?")])

    result = agent._supervisor_node(state)

    assert result["active_agent"] == "end"


def test_supervisor_ends_immediately_during_human_takeover(temp_db_path):
    agent = make_agent(temp_db_path)
    state = base_state(is_human_takeover=True, messages=[HumanMessage(content="still there?")])

    result = agent._supervisor_node(state)

    assert result["active_agent"] == "end"


def test_customer_identified_before_email_is_masked(monkeypatch):
    """The email is scrubbed from the stored message, but the customer
    lookup must still happen on the raw text."""
    import customer_support_agent as csa
    agent = csa.CustomerSupportAgent.__new__(csa.CustomerSupportAgent)

    class FakeDB:
        def get_customer_by_email(self, email):
            assert email == "alice@example.com"
            return {"customer_id": "C1", "name": "Alice Johnson", "tier": "premium"}

    agent.db = FakeDB()
    state = {"customer_id": None, "customer_tier": "standard"}
    agent._identify_from_raw(state, "My email is alice@example.com")
    assert state["customer_id"] == "C1"
    assert state["customer_tier"] == "premium"
    assert "[EMAIL_MASKED]" in agent._scrub_pii("My email is alice@example.com")
