from nemoguardrails.actions import action
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

@action(is_system_action=True, name="scrub_pii")
async def scrub_pii(context: dict):
    """Intercepts the user input and redacts PII before the LLM sees it."""
    
    # Retrieve the active message from NeMo's context (Fix for the NoneType error)
    user_message = context.get("user_message", "")
    
    # Safety check: if it's empty or None, just pass it through
    if not user_message:
        return True
        
    results = analyzer.analyze(text=user_message, entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"], language="en")
    
    if results:
        anonymized = anonymizer.anonymize(
            text=user_message,
            analyzer_results=results,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
        )
        # Overwrite the message in context with the clean version so the LLM sees the redacted text
        context["user_message"] = anonymized.text
        
    return True

# Note: The check_policy_violation action has been removed from this file 
# and is now dynamically registered inside guardrail_agent.py to resolve context errors.