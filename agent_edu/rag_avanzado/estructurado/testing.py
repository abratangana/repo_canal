from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///C:/Users/Abraham Pérez/OneDrive/Desktop/Trabajo/panomada/repositorio_canal/agent_edu/rag_avanzado/estructurado/Chinook.db")## Definimos nuestra base de datos
print(db.get_table_info()) ##Mostramos la tabla de la base de datos