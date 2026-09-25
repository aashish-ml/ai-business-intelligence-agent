from __future__ import annotations

from typing import Any


class AnswerSynthesizer:
    """
    Converts verified agent evidence into a concise,
    business-friendly answer.

    This development version is deterministic and does not
    require an external LLM API.
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

        # ---------------------------------------------------------
        # 1. Look for percentage-change analysis.
        # ---------------------------------------------------------
        for item in evidence:
            if item.get("source") != "percentage_change":
                continue

            data = item.get("data", {})

            previous_month = data.get("previous_month")
            current_month = data.get("current_month")
            previous_revenue = data.get("previous_revenue")
            current_revenue = data.get("current_revenue")
            percentage = data.get("percentage_change")

            if (
                previous_month is not None
                and current_month is not None
                and previous_revenue is not None
                and current_revenue is not None
                and percentage is not None
            ):
                if percentage > 0:
                    direction = "increased"
                elif percentage < 0:
                    direction = "decreased"
                else:
                    direction = "remained unchanged"

                return (
                    f"Sales did not decrease in the latest month. "
                    f"Revenue {direction} by "
                    f"{abs(percentage):.2f}%, from "
                    f"₹{previous_revenue:,.2f} in {previous_month} "
                    f"to ₹{current_revenue:,.2f} in "
                    f"{current_month}."
                )

        # ---------------------------------------------------------
        # 2. Customer count.
        # ---------------------------------------------------------
        for item in evidence:
            if item.get("source") != "sql_query":
                continue

            data = item.get("data", {})
            result = data.get("result", {})
            rows = result.get("rows", [])

            if (
                rows
                and "customers" in rows[0]
                and "customer" in question.lower()
            ):
                count = rows[0]["customers"]

                return (
                    f"The database currently contains "
                    f"{count:,} customers."
                )

        # ---------------------------------------------------------
        # 3. Product performance analysis.
        # ---------------------------------------------------------
        if intent == "product_analysis":

            for item in evidence:
                if item.get("source") != "sql_query":
                    continue

                data = item.get("data", {})
                result = data.get("result", {})
                rows = result.get("rows", [])

                if not rows:
                    continue

                # Calculate profit margin for every analyzed product.
                product_metrics = []

                for row in rows:
                    revenue = float(row.get("revenue", 0))
                    profit = float(row.get("estimated_profit", 0))

                    margin = (
                        (profit / revenue) * 100
                        if revenue != 0
                        else 0.0
                    )

                    product_metrics.append(
                        {
                            "product_name": row.get(
                                "product_name",
                                "Unknown product",
                            ),
                            "category": row.get(
                                "category",
                                "Unknown category",
                            ),
                            "revenue": revenue,
                            "estimated_cost": float(
                                row.get("estimated_cost", 0)
                            ),
                            "estimated_profit": profit,
                            "margin": margin,
                        }
                    )

                # Average margin across the analyzed products.
                average_margin = (
                    sum(item["margin"] for item in product_metrics)
                    / len(product_metrics)
                )

                # Find the product with the lowest margin.
                lowest_margin_product = min(
                    product_metrics,
                    key=lambda item: item["margin"],
                )

                margin_gap = (
                    average_margin
                    - lowest_margin_product["margin"]
                )

                return (
                    f"The most margin-challenged product in the "
                    f"analyzed set is "
                    f"{lowest_margin_product['product_name']} "
                    f"({lowest_margin_product['category']}). "
                    f"It has an estimated profit margin of "
                    f"{lowest_margin_product['margin']:.2f}%, "
                    f"compared with an analyzed-product average "
                    f"of {average_margin:.2f}%. "
                    f"This is approximately "
                    f"{margin_gap:.2f} percentage points below "
                    f"the average."
                )    


        
        # ---------------------------------------------------------
        # 4. Generic SQL result.
        # ---------------------------------------------------------
        for item in evidence:
            if item.get("source") != "sql_query":
                continue

            data = item.get("data", {})
            result = data.get("result", {})
            rows = result.get("rows", [])

            if rows:
                return self._format_sql_result(rows)

        # ---------------------------------------------------------
        # 5. Customer risk / churn prediction.
        # ---------------------------------------------------------
        if intent == "customer_risk_prediction":
            for item in evidence:
                if item.get("source") != "customer_risk_prediction":
                    continue

                data = item.get("data", {})

                # Support both direct tool results and wrapped results.
                result = data.get("result", data)

                if not result.get("success"):
                    return (
                        "Customer risk prediction could not be completed."
                    )

                customer_id = result.get("customer_id")
                risk_probability = result.get("risk_probability")
                risk_level = result.get("risk_level")
                threshold = result.get("decision_threshold")

                if (
                    customer_id is None
                    or risk_probability is None
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
                    f"Customer {customer_id} is currently classified as "
                    f"{risk_level} risk, with a predicted risk probability "
                    f"of {risk_probability * 100:.2f}%. "
                    f"The model decision threshold is "
                    f"{threshold_text}."
                )

        # ---------------------------------------------------------
        # Customer risk + policy multi-step answer.
        # ---------------------------------------------------------
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
                    "Customer risk prediction could not be completed."
                )

            if not rag_result:
                return (
                    "Customer risk was predicted, but the "
                    "relevant business policy could not be retrieved."
                )

            customer_id = risk_result.get("customer_id")
            risk_probability = risk_result.get(
                "risk_probability"
            )
            risk_level = risk_result.get("risk_level")

            results = rag_result.get("results", [])

            if not results:
                return (
                    f"Customer {customer_id} has a predicted risk "
                    f"probability of {risk_probability * 100:.2f}% "
                    f"({risk_level} risk), but no relevant policy "
                    f"evidence was found."
                )

            # -------------------------------------------------
            # Select policy section matching the predicted
            # risk level instead of blindly using top result.
            # -------------------------------------------------
            expected_section = (
                f"{risk_level.capitalize()}-Risk Customers"
            )

            policy = None

            for candidate in results:
                section = candidate.get("section", "")

                if section.lower() == expected_section.lower():
                    policy = candidate
                    break

            # Fallback to the highest-ranked result only if
            # the matching risk-specific policy section was
            # not retrieved.
            if policy is None:
                policy = results[0]

            return (
                f"Customer {customer_id} has a predicted risk "
                f"probability of {risk_probability * 100:.2f}%, "
                f"which is classified as {risk_level} risk. "
                f"According to {policy['filename']} "
                f"({policy['section']}), the recommended approach is: "
                f"{policy['text']}"
            )

        # ---------------------------------------------------------
        # 5. RAG / knowledge search result.
        # ---------------------------------------------------------
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

                top_result = results[0]

                filename = top_result.get(
                    "filename",
                    "business knowledge",
                )

                section = top_result.get(
                    "section",
                    "relevant section",
                )

                text = top_result.get(
                    "text",
                    "",
                )

                return (
                    f"According to {filename}, "
                    f"under the '{section}' section: "
                    f"{text}"
                )

        # ---------------------------------------------------------
        # 5. Fallback.
        # ---------------------------------------------------------
        return (
            "The analysis completed successfully, but a concise "
            "business summary could not be generated."
        )

    def _format_sql_result(
        self,
        rows: list[dict[str, Any]],
    ) -> str:

        if len(rows) == 1:
            row = rows[0]

            parts = [
                f"{key}: {value}"
                for key, value in row.items()
            ]

            return "The analysis returned: " + ", ".join(parts) + "."

        return (
            f"The analysis returned {len(rows)} rows "
            "of business data."
        )