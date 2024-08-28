from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack import Pipeline

# Prepare prompt_builder component
prompt_template = """
You are a kind assistant and you are here to help people to find the information they need.
If you don't know the answer, simply say, "I don't know".

Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(template=prompt_template)

# Prepare llm component
model_name = "gpt-4o-mini"
llm = OpenAIGenerator(model=model_name)

# Prepare the Pipeline
chat_pipeline = Pipeline()

# Add the components to the pipeline
chat_pipeline.add_component("prompt_builder", prompt_builder)
chat_pipeline.add_component("llm", llm)

# Make the connections between components in the pipeline
# chat_pipeline.connect("prompt_builder.prompt", "llm.prompt")
chat_pipeline.connect("prompt_builder", "llm")

# Our pipeline is READY!

# Let's test it
print(
    "Hi, I am a simple chatbot that doesn't remember anything. You can ask me anything, I won't even remember it again!\nType 'q' to quit.\n")
while True:
    question = input("User: ")  # Get the question from the user
    if question == "q":
        break
    result = chat_pipeline.run({"prompt_builder": {"question": question}})  # Run the pipeline
    print("Assistant: ", result["llm"]["replies"][0])  # Print the answer
print("Goodbye!")
