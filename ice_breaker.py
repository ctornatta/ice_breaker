from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile

# information = """
# Sir Patrick Stewart OBE (born 13 July 1940) is an English actor whose career has spanned seven decades in theatre, film, television, and video games. He has been nominated for Olivier, Tony, Golden Globe, Emmy, and Screen Actors Guild Awards. He received a star on the Hollywood Walk of Fame in 1996, and was knighted by Queen Elizabeth II for services to drama in 2010.
# """

if __name__ == "__main__":
    print("Hello, LangChain!")

    summary_template = """
    given the information {information} about a person from I want you to create:
    1. a short summary
    2. two interesting facts about them
"""
summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=summary_prompt_template)

linkedin_profile_url = "https://gist.githubusercontent.com/ctornatta/16667337aa9f292a3178e8cef8786f78/raw/ca2dff629ec6876b5ab2ec7491b7e5f240459424/eden-marco.json"
information = scrape_linkedin_profile(linkedin_profile_url)

print(chain.run(information=information))
