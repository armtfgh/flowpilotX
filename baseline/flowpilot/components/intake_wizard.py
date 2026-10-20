"""Streamlit wizard for FlowPilot standardized intake."""

from __future__ import annotations

import json

import streamlit as st

from components.inventory_selector import render_inventory_selector
from flora_translate.intake_agent import IntakeAgent
from flora_translate.inventory_profiles import InventoryProfile
from flora_translate.inventory_resolution import bind_inventory
from flora_translate.schemas import DesignInputPackage, IntakeAnswer


def render_intake_wizard(key_prefix: str = "flowpilot_intake") -> DesignInputPackage | None:
    """Render the blocking intake wizard and return the current package."""

    agent = IntakeAgent()
    package_key = f"{key_prefix}_package"

    st.subheader("Standardized Intake")
    st.caption("Complete the reproducible intake package before running design.")

    inventory_profile = render_inventory_selector(f"{key_prefix}_inventory")

    c1, c2 = st.columns([4, 1])
    with c1:
        use_llm = st.checkbox(
            "Use LLM field extraction",
            value=True,
            key=f"{key_prefix}_use_llm_v3",
            help=(
                "The LLM extracts stated protocol fields; fixed question IDs "
                "and readiness logic remain deterministic."
            ),
        )
        st.caption("Question IDs and blocking readiness are deterministic and reproducible.")
    with c2:
        if st.button("Reset intake", key=f"{key_prefix}_reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                if str(key).startswith(key_prefix):
                    del st.session_state[key]
            st.rerun()

    protocol = st.text_area(
        "Initial batch protocol",
        height=180,
        key=f"{key_prefix}_protocol",
        placeholder=(
            "Paste the batch protocol here. The intake agent will extract stated "
            "facts and ask standardized follow-up questions."
        ),
    )

    if st.button(
        "Analyze Intake",
        type="primary",
        disabled=not bool(protocol.strip()),
        key=f"{key_prefix}_analyze",
        use_container_width=True,
    ):
        with st.spinner("Analyzing intake..."):
            package = agent.analyze(
                protocol,
                existing_package=st.session_state.get(package_key),
                use_llm=use_llm,
            )
            if inventory_profile is not None:
                package = _apply_inventory_profile(
                    agent,
                    protocol,
                    package,
                    inventory_profile,
                )
            st.session_state[package_key] = package.model_dump()
        st.rerun()

    package = _load_package(st.session_state.get(package_key))
    if package is None:
        st.info("Start by analyzing the batch protocol.")
        return None

    if inventory_profile is not None and _profile_changed(package, inventory_profile):
        package = _apply_inventory_profile(
            agent,
            protocol,
            package,
            inventory_profile,
        )
        st.session_state[package_key] = package.model_dump()

    # Refresh sessions created before equipment review was introduced.
    package = agent.analyze(existing_package=package, use_llm=False)
    st.session_state[package_key] = package.model_dump()
    if protocol.strip() != package.raw_protocol.strip():
        st.warning("The protocol has changed. Analyze Intake again before design.")
        return None

    pending = agent.pending_questions(package)
    if pending:
        st.markdown("#### Missing fixed and conditional inputs")
        new_answers: list[IntakeAnswer] = []
        for question in pending:
            with st.expander(
                f"{question.question_id} - {question.section}",
                expanded=True,
            ):
                st.markdown(question.question)
                if question.why_needed:
                    st.caption(question.why_needed)
                if question.expected_format:
                    st.caption(f"Expected format: {question.expected_format}")
                if question.decision_impact:
                    st.caption(f"Design impact: {question.decision_impact}")
                st.caption(f"Question origin: {question.origin}")

                allow_unavailable = question.allow_unavailable
                unavailable = False
                if allow_unavailable:
                    unavailable = st.checkbox(
                        "Mark unavailable",
                        key=f"{key_prefix}_{question.question_id}_unavailable",
                    )
                answer_text = st.text_area(
                    "Answer",
                    height=90,
                    disabled=unavailable,
                    key=f"{key_prefix}_{question.question_id}_answer",
                )
                if unavailable:
                    new_answers.append(
                        IntakeAnswer(question_id=question.question_id, status="unavailable")
                    )
                elif answer_text.strip():
                    new_answers.append(
                        IntakeAnswer(
                            question_id=question.question_id,
                            answer=answer_text.strip(),
                            status="answered",
                        )
                    )

        if st.button(
            "Save Intake Answers",
            disabled=not bool(new_answers),
            key=f"{key_prefix}_save_answers",
            use_container_width=True,
        ):
            st.session_state[package_key] = agent.analyze(
                protocol,
                existing_package=package,
                answers=new_answers,
                use_llm=False,
            ).model_dump()
            st.rerun()

    package = _load_package(st.session_state.get(package_key))
    if package is None:
        return None

    _render_equipment_review(package, key_prefix)
    if package.ready_for_design:
        st.success("Intake package is ready for design.")
    else:
        st.warning(
            "Complete the protocol answers and equipment review before numerical design. "
            + ", ".join(package.missing_question_ids)
        )

    with st.expander("Frozen DesignInputPackage", expanded=False):
        st.caption(
            f"Question bank: {package.question_bank_version} · "
            f"set hash: {package.question_set_hash or 'not generated'}"
        )
        if package.active_domains:
            st.caption("Detected domains: " + ", ".join(package.active_domains))
        st.json(package.model_dump())

    return package


def _load_package(value) -> DesignInputPackage | None:
    if value is None:
        return None
    if isinstance(value, DesignInputPackage):
        return value
    return DesignInputPackage.model_validate(value)


def _apply_inventory_profile(
    agent: IntakeAgent,
    protocol: str,
    package: DesignInputPackage,
    profile: InventoryProfile,
) -> DesignInputPackage:
    return bind_inventory(package, profile)


def _render_equipment_review(package: DesignInputPackage, key_prefix: str) -> None:
    from flora_translate.inventory_profiles import inventory_profile_from_payload, save_inventory_profile
    from flora_translate.inventory_resolution import alternative_profiles, confirm_inventory

    review = package.inventory_review
    if not review.get("requirements"):
        return
    st.markdown("#### Equipment requirements")
    st.caption(f"{review.get('profile_name') or 'Intake inventory'} | version {review.get('profile_version') or 'unsaved'}")
    st.dataframe([
        {"Capability": item["category"], "Needed": item["required_count"],
         "Available": item["available_count"], "Status": item["status"]}
        for item in review["requirements"]
    ], hide_index=True, use_container_width=True)
    if not review["ready"]:
        for alternative in alternative_profiles(package):
            if st.button(f"Select {alternative['name']} v{alternative['version']}", key=f"{key_prefix}_alt_{alternative['profile_id']}"):
                st.session_state[f"{key_prefix}_inventory_pending_profile"] = alternative["profile_id"]
                st.rerun()
    for question in [*review.get("questions", []), *review.get("confirmations", [])]:
        prefix = f"{key_prefix}_{question['question_id']}_{review['input_sha256'][:12]}"
        with st.expander(f"{question['question_id']} - {question['title']}", expanded=question in review.get("questions", [])):
            st.write(question["reason"])
            status = st.radio("Equipment availability", ["available", "unavailable"], key=f"{prefix}_status")
            with st.form(f"{prefix}_form"):
                equipment = {}
                if status == "available":
                    fields = [
                        {"key": "equipment_id", "label": "Equipment ID"},
                        {"key": "name", "label": "Equipment name"},
                        {"key": "quantity", "label": "Quantity available"},
                        *question["fields"],
                    ]
                    for field in fields:
                        equipment[field["key"]] = st.text_input(field["label"], value="1" if field["key"] == "quantity" else "", key=f"{prefix}_{field['key']}")
                note = st.text_input("Confirmation note or specification source", key=f"{prefix}_note")
                submitted = st.form_submit_button("Save inventory confirmation")
            if submitted:
                try:
                    if not package.inventory_profile_snapshot:
                        raise ValueError("Select or import an inventory profile first")
                    profile = inventory_profile_from_payload(package.inventory_profile_snapshot)
                    updated, changed = confirm_inventory(profile, category=question["category"], status=status, equipment=equipment, note=note)
                    bind_inventory(package, updated)
                    if changed:
                        updated, _ = save_inventory_profile(updated)
                    st.session_state[f"{key_prefix}_package"] = bind_inventory(package, updated).model_dump()
                    st.session_state[f"{key_prefix}_inventory_pending_profile"] = updated.profile_id
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def _profile_changed(
    package: DesignInputPackage,
    profile: InventoryProfile,
) -> bool:
    current = package.inventory_profile_snapshot or {}
    return json.dumps(current, sort_keys=True, default=str) != json.dumps(
        profile.model_dump(), sort_keys=True, default=str
    )
