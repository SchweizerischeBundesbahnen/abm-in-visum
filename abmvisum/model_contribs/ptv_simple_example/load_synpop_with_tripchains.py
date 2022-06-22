import abm
from src import abm_utilities

# start logging engine
abm_utilities.start_logging()

Visum=globals()['Visum']
my_abm = abm.ABM(Visum=Visum, model_dir=r'.\\')
my_abm.load_synpop_with_tripchains()
