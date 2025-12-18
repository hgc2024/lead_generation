import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Jurist:
    def __init__(self, name: str, role_description: str, evaluation_criteria: str, model: str = "mistral"):
        self.name = name
        self.role_description = role_description
        self.evaluation_criteria = evaluation_criteria
        self.model = model
        self.llm = ChatOllama(model=model, temperature=0.1) # Low temp for consistent judging

    def evaluate(self, email_draft: str, lead_context: dict, product_context: str) -> dict:
        template = """
        Role: {role_description}
        
        Task: You are acting as a stakeholder reviewing an AI-generated sales email.
        
        Context:
        - Lead Source: {lead_source}
        - Lead Tags: {lead_tags}
        - Product Being Pitched: {product}
        
        Email Draft to Review:
        "
        {email_draft}
        "
        
        Evaluation Criteria:
        {evaluation_criteria}
        
        Output Format:
        Return ONLY a JSON object with the following fields:
        {{
            "score": <int 1-10>,
            "reasoning": "<short explanation>"
        }}
        """
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response_str = chain.invoke({
                "role_description": self.role_description,
                "lead_source": lead_context.get('Lead Source', 'Unknown'),
                "lead_tags": lead_context.get('Tags', 'None'),
                "product": product_context,
                "email_draft": email_draft,
                "evaluation_criteria": self.evaluation_criteria
            })
            
            # Clean up response to ensure typical JSON issues (like backticks) are handled
            cleaned_response = response_str.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_response)
        except Exception as e:
            return {"score": 0, "reasoning": f"Evaluation Failed: {str(e)}"}

# Define the Corporate Hierarchy / Jury
SALES_Director = Jurist(
    name="Sales Director",
    role_description="You are the VP of Sales at Western Digital. You care about conversion, persuasion, and clear Call-to-Actions (CTA).",
    evaluation_criteria="Rate the email 1-10 based on how likely it is to get a meeting. Does it show value? Is the CTA clear? Is it too long?"
)

BRAND_MANAGER = Jurist(
    name="Brand Manager",
    role_description="You are the Brand Manager. You care about tone, professionalism, and voice.",
    evaluation_criteria="Rate the email 1-10 based on professionalism. Is the tone appropriate for a premium B2B tech company? Is it too pushy or too casual?"
)

COMPLIANCE_OFFICER = Jurist(
    name="Compliance Officer",
    role_description="You are the Legal & Compliance Officer. You care strictly about factual accuracy and avoiding liability.",
    evaluation_criteria="Rate the email 1-10 based on factual accuracy. Does the email invent features not mentioned in the product name? (e.g. promising 'Cloud features' for a simple drive if not applicable). 10 = Safe/Accurate, 1 = Hallucination."
)

JURY_PANEL = [SALES_Director, BRAND_MANAGER, COMPLIANCE_OFFICER]
