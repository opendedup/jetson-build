import vertexai

from vertexai.preview.generative_models import grounding
from vertexai.generative_models import GenerationConfig, GenerativeModel, Tool

# TODO(developer): Update and un-comment below line
# project_id = "PROJECT_ID"

vertexai.init(project="lemmingsinthewind", location="us-central1")



# Use Google Search for grounding
tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval(disable_attribution=False))
model = GenerativeModel(model_name= "gemini-1.5-pro",tools=[tool])
prompt = "When is the next total solar eclipse in US?"
response = model.generate_content(
    prompt,
    tools=[tool],
    generation_config=GenerationConfig(
        temperature=0.0,
    ),
)

print(response)