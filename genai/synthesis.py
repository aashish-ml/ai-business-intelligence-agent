from __future__ import annotations

from typing import Any


class AnswerSynthesizer:
    """
    Converts verified agent evidence into concise,
    business-friendly, evidence-grounded answers.

    Deterministic development implementation.
    """

    def synthesize(
        self,
        question: str,
        intent: str,
        evidence: list[dict[str, Any]],
    ) -> str:

        if not evidence:
            return (
                "I could not find sufficient evidence to answer "
                "the question."
            )

        # =========================================================
        # 1. Revenue Change
        # =========================================================

        if intent == "business_revenue_analysis":

            for item in evidence:
                if item.get("source") != "business_analysis":
                    continue

                result = item.get("data", {}).get("result", {})

                if result.get("status") != "success":
                    continue

                previous_month = result.get("previous_month")
                current_month = result.get("current_month")
                previous_revenue = result.get("previous_revenue")
                current_revenue = result.get("current_revenue")
                absolute_change = result.get("absolute_change")
                percentage_change = result.get("percentage_change")
                direction = result.get("direction")

                if previous_revenue is None or current_revenue is None:
                    continue

                direction_text = {
                    "increased": "increased",
                    "decreased": "decreased",
                    "unchanged": "remained unchanged",
                }.get(direction, "changed")

                return (
                    f"Revenue {direction_text} by "
                    f"{abs(percentage_change):.2f}% from "
                    f"₹{previous_revenue:,.2f} in "
                    f"{previous_month} to "
                    f"₹{current_revenue:,.2f} in "
                    f"{current_month}. "
                    f"The absolute change was "
                    f"₹{abs(absolute_change):,.2f}."
                )

        # =========================================================
        # 2. Revenue Trend
        # =========================================================

        if intent == "business_revenue_trend":

            rows = self._get_business_rows(evidence)

            if rows:
                first = rows[0]
                last = rows[-1]

                return (
                    f"The revenue dataset contains {len(rows)} "
                    f"monthly observations, from "
                    f"{first.get('month')} to {last.get('month')}. "
                    f"Revenue was "
                    f"₹{float(first.get('revenue', 0)):,.2f} "
                    f"in {first.get('month')} and "
                    f"₹{float(last.get('revenue', 0)):,.2f} "
                    f"in {last.get('month')}."
                )

        # =========================================================
        # 3. Category Performance
        # =========================================================

        if intent == "category_performance":

            rows = self._get_business_rows(evidence)

            if rows:
                top = rows[0]

                return (
                    f"{top.get('category', 'Unknown category')} "
                    f"generated the highest revenue among the "
                    f"analyzed categories, with "
                    f"₹{float(top.get('revenue', 0)):,.2f} "
                    f"from {int(top.get('units_sold', 0)):,} units."
                )

        # =========================================================
        # 4. Product Performance
        # =========================================================

        if intent == "product_performance":

            rows = self._get_business_rows(evidence)

            if rows:
                top = rows[0]

                return (
                    f"{top.get('product_name', 'Unknown product')} "
                    f"is the highest-revenue product in the "
                    f"analyzed set, generating "
                    f"₹{float(top.get('revenue', 0)):,.2f} "
                    f"from {int(top.get('units_sold', 0)):,} units."
                )

        # =========================================================
        # 5. Segment Performance
        # =========================================================

        if intent == "segment_performance":

            rows = self._get_business_rows(evidence)

            if rows:
                top = rows[0]

                return (
                    f"{top.get('segment_name', 'Unknown segment')} "
                    f"is the highest-revenue customer segment in "
                    f"the analyzed data, generating "
                    f"₹{float(top.get('revenue', 0)):,.2f} "
                    f"across {int(top.get('customers', 0)):,} "
                    f"customers."
                )

        # =========================================================
        # 6. Order Status
        # =========================================================

        if intent == "order_status_analysis":

            rows = self._get_business_rows(evidence)

            if rows:
                parts = []

                for row in rows:
                    parts.append(
                        f"{row.get('status', 'unknown')}: "
                        f"{int(row.get('order_count', 0)):,}"
                    )

                return (
                    "Order status distribution: "
                    + ", ".join(parts)
                    + "."
                )

        # =========================================================
        # 7. Customer Risk Prediction
        # =========================================================

        if intent == "customer_risk_prediction":

            for item in evidence:
                if item.get("source") != "customer_risk_prediction":
                    continue

                data = item.get("data", {})
                result = data.get("result", data)

                if not result.get("success"):
                    return (
                        "Customer risk prediction could not "
                        "be completed."
                    )

                customer_id = result.get("customer_id")
                probability = result.get("risk_probability")
                risk_level = result.get("risk_level")
                threshold = result.get("decision_threshold")

                if (
                    customer_id is None
                    or probability is None
                    or risk_level is None
                ):
                    return (
                        "Customer risk prediction completed, "
                        "but the result was incomplete."
                    )

                threshold_text = (
                    f"{threshold * 100:.0f}%"
                    if threshold is not None
                    else "the configured threshold"
                )

                return (
                    f"Customer {customer_id} is currently "
                    f"classified as {risk_level} risk, with a "
                    f"predicted risk probability of "
                    f"{probability * 100:.2f}%. "
                    f"The model decision threshold is "
                    f"{threshold_text}."
                )

        # =========================================================
        # 8. Customer Risk + Policy
        # =========================================================

        if intent == "customer_risk_policy":

            risk_result = None
            rag_result = None

            for item in evidence:
                source = item.get("source")
                data = item.get("data", {})
                result = data.get("result", data)

                if source == "customer_risk_prediction":
                    risk_result = result

                elif source == "rag_search":
                    rag_result = result

            if not risk_result:
                return (
                    "Customer risk prediction could not "
                    "be completed."
                )

            if not rag_result:
                return (
                    "Customer risk was predicted, but the "
                    "relevant business policy could not "
                    "be retrieved."
                )

            customer_id = risk_result.get("customer_id")
            probability = risk_result.get("risk_probability")
            risk_level = risk_result.get("risk_level")

            results = rag_result.get("results", [])

            if not results:
                return (
                    f"Customer {customer_id} has a predicted "
                    f"risk probability of "
                    f"{probability * 100:.2f}% "
                    f"({risk_level} risk), but no relevant "
                    f"policy evidence was found."
                )

            expected_section = (
                f"{risk_level.capitalize()}-Risk Customers"
            )

            policy = None

            for candidate in results:
                if (
                    candidate.get("section", "").lower()
                    == expected_section.lower()
                ):
                    policy = candidate
                    break

            if policy is None:
                policy = results[0]

            return (
                f"Customer {customer_id} has a predicted risk "
                f"probability of {probability * 100:.2f}%, "
                f"which is classified as {risk_level} risk. "
                f"According to {policy.get('filename', 'business policy')} "
                f"({policy.get('section', 'relevant section')}), "
                f"the recommended approach is: "
                f"{policy.get('text', '')}"
            )

        # =========================================================
        # 9. Product Margin Analysis
        # =========================================================

        if intent == "product_analysis":

            for item in evidence:
                if item.get("source") != "sql_query":
                    continue

                result = item.get("data", {}).get("result", {})
                rows = result.get("rows", [])

                if not rows:
                    continue

                metrics = []

                for row in rows:
                    revenue = float(row.get("revenue", 0))
                    profit = float(
                        row.get("estimated_profit", 0)
                    )

                    margin = (
                        (profit / revenue) * 100
                        if revenue
                        else 0.0
                    )

                    metrics.append(
                        {
                            "product_name": row.get(
                                "product_name",
                                "Unknown product",
                            ),
                            "category": row.get(
                                "category",
                                "Unknown category",
                            ),
                            "margin": margin,
                        }
                    )

                average_margin = (
                    sum(x["margin"] for x in metrics)
                    / len(metrics)
                )

                lowest = min(
                    metrics,
                    key=lambda x: x["margin"],
                )

                return (
                    f"The most margin-challenged product in the "
                    f"analyzed set is "
                    f"{lowest['product_name']} "
                    f"({lowest['category']}). "
                    f"It has an estimated profit margin of "
                    f"{lowest['margin']:.2f}%, compared with an "
                    f"analyzed-product average of "
                    f"{average_margin:.2f}%."
                )

        # =========================================================
        # 10. Knowledge / RAG Search
        # =========================================================

        if intent == "knowledge_search":

            for item in evidence:
                if item.get("source") != "rag_search":
                    continue

                data = item.get("data", {})
                result = data.get("result", data)

                if not result.get("success"):
                    return (
                        "Knowledge search could not be completed."
                    )

                results = result.get("results", [])

                if not results:
                    return (
                        "No relevant business policy or "
                        "knowledge was found."
                    )

                top = results[0]

                return (
                    f"According to "
                    f"{top.get('filename', 'business knowledge')}, "
                    f"under the "
                    f"'{top.get('section', 'relevant section')}' "
                    f"section: "
                    f"{top.get('text', '')}"
                )

        # =========================================================
        # 11. Generic SQL Result
        # =========================================================

        for item in evidence:
            if item.get("source") != "sql_query":
                continue

            result = item.get("data", {}).get("result", {})
            rows = result.get("rows", [])

            if rows:
                return self._format_sql_result(rows)

        return (
            "The analysis completed successfully, but a concise "
            "business summary could not be generated."
        )

    # =============================================================
    # Helpers
    # =============================================================

    @staticmethod
    def _get_business_rows(
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        for item in evidence:

            if item.get("source") != "business_analysis":
                continue

            data = item.get("data", {})
            result = data.get("result", {})

            # Some business-analysis metrics return:
            # {
            #     "data": [...]
            # }
            #
            # while other metrics return:
            # {
            #     "status": "success",
            #     ...
            # }
            #
            # Support both formats.

            rows = result.get("data")

            if isinstance(rows, list):
                return rows

            if result.get("status") == "success":
                rows = result.get("rows", [])

                if isinstance(rows, list):
                    return rows

        return []

    @staticmethod
    def _format_sql_result(
        rows: list[dict[str, Any]],
    ) -> str:

        if len(rows) == 1:

            parts = [
                f"{key}: {value}"
                for key, value in rows[0].items()
            ]

            return (
                "The analysis returned: "
                + ", ".join(parts)
                + "."
            )

        return (
            f"The analysis returned {len(rows)} rows "
            "of business data."
        )