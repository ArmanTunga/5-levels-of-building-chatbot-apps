from typing import List

from components import CustomSQLRetriever
from dotenv import load_dotenv
load_dotenv()

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.routers import ConditionalRouter
from haystack.components.generators import OpenAIGenerator

template="""
    Generate a SQL Query for a PostgreSQL database based on the given schema and question.
    
    Schema:
        Tables:
        1. listings (listing_id, name, city, property_type, amenities, price, minimum_nights, review_scores_rating, instant_bookable, host_id)
        2. hosts (host_id, host_name, host_response_rate, host_verifications, is_superhost, response_time)
    
        Notes:
        - The 'amenities' column in the 'listings' table is a text field that can contain multiple amenities.
        - The 'host_verifications' column in the 'hosts' table is a text field that can contain multiple verifications.

    Instructions:
        1. Do not include any explanations or apologies in your responses.
        2: Do not respond to any questions that might ask anything other than for you to construct a SQL statement. 
        3. Do not include any text except the generated SQL query.
        4. Use only the provided column names in the schema.
        5. Do not run any queries that would add to or delete from the database.
        6. If you need to divide numbers, make sure to filter the denominator to be non-zero.
        7. Use IS NULL or IS NOT NULL when analyzing missing properties.
        8. Never return embedding properties in your queries.
        9. Do not include new line in the query. Write the sql query in one line. 
        10 Always use LIMIT 5 creating a query.
        11. If user's query is not related to the schema, return "not related I can only provide information about the listings and hosts."
        12. If the query cannot be answered with the schema, return "not related I don't have the information for your question."
    
    Examples:     
    
    Question: Can you show me listings of superhosts in Paris?
    Answer: SELECT * FROM listings l JOIN hosts h ON l.host_id = h.host_id WHERE l.city = 'Paris' AND h.is_superhost = TRUE LIMIT 5;
    
    Question: What are the top 5 most expensive listings in Paris with a review score above 4.5?
    Answer: SELECT name, price, review_scores_rating FROM listings WHERE city = 'Paris' AND review_scores_rating > 4.5 ORDER BY price DESC LIMIT 5;
    
    Question: How many listings are there for each property type in Paris?
    Answer: SELECT property_type, COUNT(*) as listing_count FROM listings WHERE city = 'Paris' GROUP BY property_type ORDER BY listing_count DESC LIMIT 5;
    
    Question: What is the average price of listings in Paris?
    Answer: SELECT AVG(l.price) as avg_price FROM listings l LIMIT 5;
    
    Question: Which hosts have the fastest response time?
    Answer: SELECT host_name, response_time FROM hosts WHERE response_time IS NOT NULL ORDER BY  CASE  WHEN response_time = 'within fifteen minutes' THEN 1 WHEN response_time = 'within half an hour' THEN 2 WHEN response_time = 'within an hour' THEN 3 WHEN response_time = 'more than one hour' THEN 4 ELSE 5 END LIMIT 5;
    
    Question: What are the best foods in Paris?
    Answer: not related I can only provide information about the listings and hosts.
    
    Question: Can you please list me all the listings close to Eiffel?
    Answer: not related I don't have the information for your question.
    
    
    User's Question: {{ question }}
    Answer:
    """

prompt = PromptBuilder(template=template)

model_name= "gpt-4o-mini"
text_2_sql_llm = OpenAIGenerator(model=model_name)

# Also we will have a conditional router component.
# router = ConditionalRouter(routes)

custom_sql_retriever = CustomSQLRetriever()

system_prompt = """You are a helpful and friendly assistant that helps people with their questions about the listings and hosts.
From given information about listings or hosts, generate a helpful, conversational response to the user's question in a friendly tone.
If there is no information, simply say "I don't have the information for your question"
"""
final_llm = OpenAIGenerator(model=model_name, system_prompt=system_prompt) # or you can add another prompt builder before this llm instead of using system_prompt. That way you can use only 1 llm.


routes = [
     {
        "condition": "{{'not related' not in replies[0]|lower}}",
        "output": "{{replies[0]}}",
        "output_name": "sql_query",
        "output_type": str,
    },
    {
        "condition": "{{'not related' in replies[0]|lower}}",
        "output": "{{replies[0].replace('not related', '')|trim}}",
        "output_name": "not_related_answer",
        "output_type": str,
    },
]

router = ConditionalRouter(routes)

router.run(replies=["SELECT * FROM listings;"], query="Can you show me all the listings?")
router.run(replies=["not related I don't have the information for your question."], query="Can you please list me all the listings close to Eiffel?")

pipeline = Pipeline()
pipeline.add_component("prompt_builder", prompt)
pipeline.add_component("text_to_sql", text_2_sql_llm)
pipeline.add_component("router", router)
pipeline.add_component("custom_sql_retriever", custom_sql_retriever)
pipeline.add_component("final_llm", final_llm)

pipeline.connect("prompt_builder", "text_to_sql")
pipeline.connect("text_to_sql", "router")
pipeline.connect("router.sql_query", "custom_sql_retriever")
pipeline.connect("custom_sql_retriever", "final_llm")

from pathlib import Path
pipeline.draw(Path("images/level_3/pipeline.png"))

user_input = "Can you show me all the listings?"
result = pipeline.run({"prompt_builder": {"question": user_input}})
final_llm_result = result.get("final_llm", None)
if final_llm_result:
    print(f"Assistant: {final_llm_result["replies"][0]}")
else:
    print(f"Assistant: {result["router"]["not_related_answer"]}")

user_input = "Can you tell me where I can find the Louvre?"
result = pipeline.run({"prompt_builder": {"question": user_input}})
final_llm_result = result.get("final_llm", None)
if final_llm_result:
    print(f"Assistant: {final_llm_result["replies"][0]}")
else:
    print(f"Assistant: {result["router"]["not_related_answer"]}")