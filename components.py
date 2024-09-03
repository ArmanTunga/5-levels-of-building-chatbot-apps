from dotenv import load_dotenv
load_dotenv()
import os
import psycopg2
import psycopg2.extras
from haystack import component

@component
class CustomSQLRetriever:
    """
    Component for executing the given SQL Query and returning results for Postgres.
    """

    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )

    @component.output_types(query_results=str)
    def run(self, sql_query: str) -> dict[str, str] | dict[str, str]:
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql_query)
                results = cur.fetchall()
                results_str = ""
                for result in results:
                    result_dict = dict(result)
                    del result_dict['listing_id']
                    results_str += f"{result_dict}\n"

                return {"query_results": results_str}
        except Exception as e:
            print("An error occurred while executing the SQL Query: ", e)
            return {"query_results": ""}

    def close(self):
        if self.conn:
            self.conn.close()





if __name__ == "__main__":
    retriever = CustomSQLRetriever()
    query = "SELECT * FROM listings l JOIN hosts h ON l.host_id = h.host_id WHERE l.city = 'Paris' AND h.is_superhost = TRUE LIMIT 3;"
    results = retriever.run(query)
    print(results["query_results"])
    retriever.close()