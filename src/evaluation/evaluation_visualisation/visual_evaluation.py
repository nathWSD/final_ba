import json
import os
import pymysql
import datetime
import traceback 
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)

MYSQL_HOST = os.getenv('MYSQL_HOST')

MYSQL_PORT = int(os.getenv('MYSQL_PORT'))
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

GRAFANA_IMAGE_NAME = os.getenv('GRAFANA_IMAGE_NAME')
GRAFANA_CONTAINER_NAME = os.getenv('GRAFANA_CONTAINER_NAME')
GRAFANA_VOLUME_NAME = os.getenv('GRAFANA_VOLUME_NAME') 
GRAFANA_HOST_PORT = int(os.getenv('GRAFANA_HOST_PORT')) 

JSON_FILE_FOLDER = os.path.join(os.getcwd(), 'src/evaluation/pipeline_evaluation_data')

def load_json_to_mysql(json_path, shared_start_time):
    """
    Reads JSON data, adds timestamps, ensures target database exists,
    and inserts into a MySQL table using PyMySQL.
    """

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at '{json_path}'")
        return False

    table_name = os.path.splitext(os.path.basename(json_path))[0]
    print(f"Target MySQL table name: '{table_name}' in database '{MYSQL_DATABASE}'")

    conn = None
    try:
        # --- Step 1: Connect without specifying database initially ---
        print(f"Connecting to MySQL Server: host={MYSQL_HOST}, port={MYSQL_PORT}, user={MYSQL_USER}")
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            connect_timeout=15,
            cursorclass=pymysql.cursors.DictCursor 
        )
        print("Successfully connected to MySQL server.")

        # --- Step 2: Ensure the target database exists ---
        with conn.cursor() as cursor:
            print(f"Ensuring database '{MYSQL_DATABASE}' exists...")
            # Use backticks `` for safety around database name
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`")
            print(f"Database '{MYSQL_DATABASE}' check/creation complete.")
            # Select the database for the rest of the session
            cursor.execute(f"USE `{MYSQL_DATABASE}`")
            print(f"Using database '{MYSQL_DATABASE}'.")

        # --- Step 3: Proceed with table creation and data insertion ---
        with conn.cursor() as cursor: # Use the same connection
            # Create table if it doesn't exist
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `search_type` VARCHAR(150),                     -- Increased length slightly
                `precision` FLOAT,
                `recall` FLOAT,
                `relevancy` FLOAT,
                `rouge1` FLOAT,                                 -- New metric
                `cosine_similarity` FLOAT,                      -- New metric
                `num_input_token` INT,
                `num_output_token` INT,
                `time_taken` FLOAT,
                `query` MEDIUMTEXT,                                   -- New text field
                `actual_output` MEDIUMTEXT,                           -- New text field
                `expected_output` MEDIUMTEXT,                         -- New text field
                `retrieval_context` MEDIUMTEXT,               
                `primary_llm_used` VARCHAR(100),                -- New model info
                `fallback_llm_attempted` VARCHAR(100) NULL,     -- New fallback info (allow NULL)
                `fallback_used_and_succeeded_relevancy` BOOLEAN, -- New flag
                `fallback_used_and_succeeded_precision` BOOLEAN, -- New flag
                `fallback_used_and_succeeded_recall` BOOLEAN,    -- New flag
                `final_state_failed_relevancy` BOOLEAN,          -- New flag
                `final_state_failed_precision` BOOLEAN,          -- New flag
                `final_state_failed_recall` BOOLEAN,             -- New flag
                `created_at` TIMESTAMP NULL                     -- Original timestamp
            ) ENGINE=InnoDB CHARACTER SET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            print(f"Executing: CREATE TABLE IF NOT EXISTS `{table_name}`...")
            cursor.execute(create_table_sql)
            print("Table check/creation complete.")

            # Read JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                print("Error: JSON file should contain a list of objects.")
                conn.close()
                return False

            # Prepare insert statement
            insert_sql = f"""
            INSERT INTO `{table_name}` (
                search_type, `precision`, recall, relevancy, rouge1, cosine_similarity,
                num_input_token, num_output_token, time_taken,
                `query`, actual_output, expected_output, retrieval_context,
                primary_llm_used, fallback_llm_attempted,
                fallback_used_and_succeeded_relevancy, fallback_used_and_succeeded_precision, fallback_used_and_succeeded_recall,
                final_state_failed_relevancy, final_state_failed_precision, final_state_failed_recall,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            rows_to_insert = []
            start_time = shared_start_time

            for i, entry in enumerate(data):
                current_timestamp = start_time + datetime.timedelta(seconds=i)
                rows_to_insert.append((
                    entry.get("search_type"),
                    entry.get("precision"),
                    entry.get("recall"),
                    entry.get("relevancy"),
                    entry.get("rouge1"),                                 
                    entry.get("cosine_similarity"),                      
                    entry.get("num_input_token"),
                    entry.get("num_output_token"),
                    entry.get("time_taken"),
                    entry.get("query"),                                  
                    entry.get("actual_output"),                          
                    entry.get("expected_output"),                        
                    entry.get("retrieval_context"),                                         
                    entry.get("primary_llm_used"),                       
                    entry.get("fallback_llm_attempted"),                 
                    bool(entry.get("fallback_used_and_succeeded_relevancy", False)),
                    bool(entry.get("fallback_used_and_succeeded_precision", False)),
                    bool(entry.get("fallback_used_and_succeeded_recall", False)),   
                    bool(entry.get("final_state_failed_relevancy", False)),         
                    bool(entry.get("final_state_failed_precision", False)),         
                    bool(entry.get("final_state_failed_recall", False)),           
                    current_timestamp.strftime('%Y-%m-%d %H:%M:%S') 
                ))

            # Insert data if any
            if rows_to_insert:
                print(f"Inserting {len(rows_to_insert)} rows into `{table_name}`...")
                rowcount = cursor.executemany(insert_sql, rows_to_insert)
                conn.commit() # Commit the transaction
                print(f"{rowcount} rows inserted successfully.")
            else:
                print("No data found in JSON to insert.")

        # --- Step 4: Close connection ---
        conn.close()
        print(f"PyMySQL connection closed for {table_name}.")
        return True

    # --- Error Handling ---
    except pymysql.Error as err:
        error_code = err.args[0] if err.args else 'N/A'
        print(f"PyMySQL Error (Code: {error_code}): {err}")
        if error_code == 1044: print(f"Hint: User '{MYSQL_USER}' might lack permissions (e.g., CREATE DATABASE or privileges on '{MYSQL_DATABASE}').")
        elif error_code == 1045: print("Hint: Check MySQL username/password.")
        # 1049 shouldn't happen here anymore due to CREATE DATABASE IF NOT EXISTS
        elif error_code == 2003: print(f"Hint: Ensure MySQL server is running and accessible on {MYSQL_HOST}:{MYSQL_PORT}.")
        # Handle other potential errors during CREATE/INSERT
        elif error_code == 1146: print(f"Hint: Table '{table_name}' might not have been created successfully.")

        if conn: conn.close() # Ensure connection is closed on error
        return False
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_path}'")
        # No connection to close here usually
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{json_path}'.")
        if conn: conn.close()
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc() # Print full traceback for unexpected errors
        if conn: conn.close()
        return False


def ingest_graph_stats_to_sql(graph_stats_path):
    if not os.path.exists(graph_stats_path):
        print(f"Error: JSON file not found at '{graph_stats_path}'")
        return False

    table_name = os.path.splitext(os.path.basename(graph_stats_path))[0]
    print(f"Target MySQL table name: '{table_name}' in database '{MYSQL_DATABASE}'")
    conn = None

        # --- Step 1: Connect without specifying database initially ---
    print(f"Connecting to MySQL Server: host={MYSQL_HOST}, port={MYSQL_PORT}, user={MYSQL_USER}")
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        connect_timeout=15,
        cursorclass=pymysql.cursors.DictCursor 
    )
    print("Successfully connected to MySQL server.")
    # --- Step 2: Ensure the target database exists ---
    with conn.cursor() as cursor:
        print(f"Ensuring database '{MYSQL_DATABASE}' exists...")
        # Use backticks `` for safety around database name
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`")
        print(f"Database '{MYSQL_DATABASE}' check/creation complete.")
        # Select the database for the rest of the session
        cursor.execute(f"USE `{MYSQL_DATABASE}`")
        print(f"Using database '{MYSQL_DATABASE}'.")
    # --- Step 3: Proceed with table creation and data insertion ---
    with conn.cursor() as cursor: # Use the same connection
        # Create table if it doesn't exist
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `graph_type` VARCHAR(150), 
            `number_of_nodes` INT,                     
            `number_of_relationships` INT,
            `mean_distance` FLOAT,
            `diameter` FLOAT,
            `radius` FLOAT,  
            `num_wcc` INT, 
            `largest_wcc_size` INT,
            `num_scc` INT, 
            `largest_scc_size` INT,            
            `communities_level_1` INT,                      
            `communities_level_2` INT,
            `communities_level_3` INT
        ) ENGINE=InnoDB CHARACTER SET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        print(f"Executing: CREATE TABLE IF NOT EXISTS `{table_name}`...")
        cursor.execute(create_table_sql)
        print("Table check/creation complete.")
        # Read JSON data
        with open(graph_stats_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("Error: JSON file should contain a list of objects.")
            conn.close()
            return False
        # Prepare insert statement
        insert_sql = f"""
        INSERT INTO `{table_name}` (
            graph_type, `number_of_nodes`, number_of_relationships, mean_distance, diameter, radius,
            num_wcc, largest_wcc_size, num_scc, largest_scc_size, communities_level_1, communities_level_2,
            `communities_level_3`
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        rows_to_insert = []
        for i, entry in enumerate(data):
            rows_to_insert.append((
                entry.get("graph_type"),
                entry.get("number_of_nodes"),
                entry.get("number_of_relationships"),
                entry.get("mean_distance"),
                entry.get("diameter"),                                 
                entry.get("radius"),                      
                entry.get("num_wcc"),
                entry.get("largest_wcc_size"),
                entry.get("num_scc"),
                entry.get("largest_scc_size"),
                entry.get("communities_level_1"),
                entry.get("communities_level_2"),
                entry.get("communities_level_3")
            ))
        # Insert data if any
        if rows_to_insert:
            print(f"Inserting {len(rows_to_insert)} rows into `{table_name}`...")
            rowcount = cursor.executemany(insert_sql, rows_to_insert)
            conn.commit() # Commit the transaction
            print(f"{rowcount} rows inserted successfully.")
        else:
            print("No data found in JSON to insert.")
    # --- Step 4: Close connection ---
    conn.close()
    print(f"PyMySQL connection closed for {table_name}.")
    return True
    
    

def grafana_visual_eval_data(graph_stats_path):
  
    print("--- Step 1: Load JSON data into MySQL ---")

    shared_start_time = datetime.datetime.now() 
           
    for filename in os.listdir(JSON_FILE_FOLDER):
            if filename.endswith('.json'):
                json_path = os.path.join(JSON_FILE_FOLDER, filename)
                data_loaded = load_json_to_mysql(json_path, shared_start_time)
                
    stats_success = ingest_graph_stats_to_sql(graph_stats_path)

    if not data_loaded or not stats_success:
            print("\n--- Halting script due to database error. Grafana will not be started. ---")
    else:
        print("\n--- Step 2: Start Grafana Docker Container ---")
        print("\n--- Next Steps in Grafana UI ---")
        print(f"1. Open Grafana: http://localhost:{GRAFANA_HOST_PORT}")
        print("2. Log in (default: admin / admin, change password).")
        print("3. Go to Configuration (gear icon) > Data Sources.")
        print("4. Click 'Add data source', search for and select 'MySQL'.")
        print("5. Configure MySQL Connection:")
        print("   - Name: Choose a name")
        print("Note: copy UID settings after sql-connection and chang the uid in json panel file to this value")
        print("************** UI Accesses *******************")
        print("QDRANT UI Access: http://localhost:6333/dashboard#/welcome")
        print("NEO4J UI Access: http://localhost:7474/browser/")
        print(f"GRAFANA UI Access: http://localhost:{GRAFANA_HOST_PORT}/")


  
  
  
  
    
    