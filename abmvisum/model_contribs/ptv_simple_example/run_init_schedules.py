import logging

import abm

from src import abm_utilities

# start logging engine
abm_utilities.start_logging()

Visum=globals()['Visum']
my_abm = abm.ABM(Visum=Visum, model_dir=r'.\\')

Visum.Graphic.StopDrawing = True

logging.info('--- initialize schedules ---')
my_abm.init_schedules()

Visum.Graphic.StopDrawing = False

logging.info('--- initialize schedules completed ---')
