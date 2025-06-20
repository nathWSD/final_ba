import nest_asyncio
try:
    nest_asyncio.apply()
except Exception as e:
    print(f"Failed to apply nest_asyncio: {e}")
from src.evaluation.evaluation_visualisation.visual_evaluation import grafana_visual_eval_data
from src.ingestion_pipeline.all_ingestion_pipeline import ingestion_complete_data
from src.query_pipeline.all_query_pipeline import retriever_and_metrics_analysis_pipeline
from src.evaluation.question_answer_generator.generate_qa import generate_eval_questions_answers
from src.ingestion_pipeline.web_site_dataset_creation import crawl_web
from src.ingestion_pipeline.helper_functions import create_neo4j_driver

import os
import asyncio
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env') , override=True)
 
urls = [
        "https://science.nasa.gov/solar-system/solar-system-facts/",
        "https://science.nasa.gov/asteroids-comets-meteors/",
        "https://science.nasa.gov/solar-system/asteroids/",
        "https://science.nasa.gov/solar-system/asteroids/2024-yr4/",
        "https://science.nasa.gov/solar-system/asteroids/16-psyche/",
        "https://science.nasa.gov/solar-system/asteroids/101955-bennu/",
        "https://science.nasa.gov/solar-system/asteroids/apophis/",
        "https://science.nasa.gov/solar-system/asteroids/dinkinesh/",
        "https://science.nasa.gov/solar-system/asteroids/didymos/",
        "https://science.nasa.gov/solar-system/asteroids/4-vesta/",
        "https://science.nasa.gov/solar-system/asteroids/433-eros/",
        "https://science.nasa.gov/solar-system/asteroids/243-ida/",
        "https://science.nasa.gov/solar-system/asteroids/25143-itokawa/",
        "https://science.nasa.gov/solar-system/comets/",
        "https://science.nasa.gov/solar-system/comets/103p-hartley-hartley-2/",
        "https://science.nasa.gov/solar-system/comets/109p-swift-tuttle/",
        "https://science.nasa.gov/solar-system/comets/19p-borrelly/",
        "https://science.nasa.gov/solar-system/comets/1p-halley/",
        "https://science.nasa.gov/solar-system/comets/67p-churyumov-gerasimenko/",
        "https://science.nasa.gov/solar-system/comets/2i-borisov/",
        "https://science.nasa.gov/solar-system/comets/2p-encke/",
        "https://science.nasa.gov/solar-system/comets/81p-wild/",
        "https://science.nasa.gov/solar-system/comets/9p-tempel-1/",
        "https://science.nasa.gov/solar-system/comets/c-1995-o1-hale-bopp/",
        "https://science.nasa.gov/solar-system/comets/c-2013-a1-siding-spring/",
        "https://science.nasa.gov/solar-system/comets/c-2012-s1-ison/",
        "https://science.nasa.gov/solar-system/comets/oumuamua/",
        "https://science.nasa.gov/solar-system/comets/p-shoemaker-levy-9/",
        "https://science.nasa.gov/solar-system/meteors-meteorites/",
        "https://science.nasa.gov/solar-system/kuiper-belt/",
        "https://science.nasa.gov/solar-system/kuiper-belt/facts/",
        "https://science.nasa.gov/solar-system/kuiper-belt/exploration/",
        "https://science.nasa.gov/sun/",
        "https://science.nasa.gov/solar-system/planets/",
        "https://science.nasa.gov/mercury/facts/",
        "https://science.nasa.gov/venus/venus-facts/",
        "https://science.nasa.gov/earth/facts/",
        "https://science.nasa.gov/mars/facts/",
        "https://science.nasa.gov/jupiter/jupiter-facts/",
        "https://science.nasa.gov/saturn/facts/",
        "https://science.nasa.gov/uranus/facts/",
        "https://science.nasa.gov/neptune/neptune-facts/",
        "https://science.nasa.gov/dwarf-planets/ceres/facts/",
        "https://science.nasa.gov/dwarf-planets/pluto/facts/",
        "https://science.nasa.gov/dwarf-planets/haumea/",
        "https://science.nasa.gov/dwarf-planets/makemake/",
        "https://science.nasa.gov/dwarf-planets/eris/"
    ]

async def main(ingest_all_data: bool, set_size:int = 34, questions_per_level: int = 2):
    
    driver = create_neo4j_driver()
    if not driver:
        print("*********** Failed to establish Neo4j connection. Check your credentials and server status.**************")
        return
        
    if ingest_all_data:
        # ingest the web content from the list of urls, clean the content and save it to txt
        dataset_path = await crawl_web.create_pipeline_dataset(urls=urls)
        # ingest the txt dataset to neo4j as graphs and qdrant as vectors
        graph_stats_path = ingestion_complete_data(driver, dataset_path)
        #after data ingestion we need to generate question from dataset with answers for evaluation    
        json_qa_path = generate_eval_questions_answers(dataset_path, set_size, questions_per_level)
    
    else:
        json_qa_path = os.path.join(os.getcwd(), 'src/evaluation/question_answer_generator/qa_data-Kopie.json')
    
    graph_stats_path = os.path.join(os.getcwd(), 'src/ingestion_pipeline/graph_analysis_stats.json')     
    # after generating the questions and answers per level use the questions in the pipeline and evaluate the answers 
    retriever_and_metrics_analysis_pipeline(driver, json_qa_path)    
    # run grafana visualization for all questions, answers and context for visualization
    #grafana_visual_eval_data(graph_stats_path)
    
    
if __name__ == "__main__":
    #NOTE before running main start all services from docker-compose.yml
    #NOTE change collection name for different datasets in the .env file
    #NOTE use netstat -ano | findstr :<PORT> then taskkill /PID <PID> /F to free a port incase it is occupied

    asyncio.run(main(ingest_all_data = False))    
