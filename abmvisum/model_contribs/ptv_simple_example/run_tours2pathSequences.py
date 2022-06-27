import collections

from src import visum_utilities

import abm

CHUNK_SIZE = 10000000

def run(Visum):

    # definition of input/output files
    pathSeqItemsAttFile = "PathSeqItems.att"

    # all trips
    trips = Visum.Net.Trips

    # list of all filtered trips
    tripAttrs = list(zip(
        visum_utilities.GetMultiByFormula(trips, r"[INDEX] = [TOUR\LAST:TRIPS\INDEX]", chunk_size=CHUNK_SIZE, reindex=True),
        # bool: isLastTripOfTour
        visum_utilities.GetMulti(trips, r'TOUR\FIRST:TRIPS\FROMACTIVITYEXECUTION\LOCATION\ZONE\NO', chunk_size=CHUNK_SIZE),
        # -> OrigZoneNo of path sequence
        visum_utilities.GetMulti(trips, r'TOUR\LAST:TRIPS\TOACTIVITYEXECUTION\LOCATION\ZONE\NO', chunk_size=CHUNK_SIZE),
        # -> DestZoneNo of path sequence
        visum_utilities.GetMulti(trips, r'INDEX', chunk_size=CHUNK_SIZE),  # -> path sequence item index
        visum_utilities.GetMulti(trips, r'FROMACTIVITYEXECUTION\LOCATION\ZONE\NO', chunk_size=CHUNK_SIZE),
        # -> ZoneNo of path sequence item (for all but last path sequence items)
        visum_utilities.GetMulti(trips, r'TOACTIVITYEXECUTION\LOCATION\ZONE\NO', chunk_size=CHUNK_SIZE),
        # is used only from the last trip -> ZoneNo of path sequence item (only for last path sequence item)
        visum_utilities.GetMulti(trips, r'DSEGCODE', chunk_size=CHUNK_SIZE),
        # -> PostDSegCode = DSegCode of the trip starting from this item
        visum_utilities.GetMulti(trips, r'TOUR\PERSONNO', chunk_size=CHUNK_SIZE),  # tour identifier
        visum_utilities.GetMulti(trips, r'TOUR\SCHEDULENO', chunk_size=CHUNK_SIZE),  # tour identifier
        visum_utilities.GetMulti(trips, r'TOURNO', chunk_size=CHUNK_SIZE),  # tour identifier
    ))

    # trips by the OD pair of the tour. will be sorted below
    # the path sequence items must be sorted by OD pair (requested by path sequence importer)
    odDict = collections.defaultdict(list)
    for tripAttr in tripAttrs:
        odDict[(tripAttr[1], tripAttr[2])].append(tripAttr)

    # header of path sequences attribute file
    attfileheader = r"""$VISION
$VERSION:VERSNR;FILETYPE;LANGUAGE;UNIT
10.00;Att;ENG;KM

$PATHSEQUENCEITEM:PATHSEQUENCESETNO;ORIGZONENO;DESTZONENO;PATHSEQUENCEINDEX;INDEX;ZONENO;POSTDSEGCODE;PATHSEQUENCE\VOL;PATHSEQUENCE\PERSONNO;PATHSEQUENCE\SCHEDULENO;PATHSEQUENCE\TOURNO
"""

    # write path sequence attribute file: one line per path sequence item. Since each start or stop location is one item, there is one item more than trips in a tour
    with open(pathSeqItemsAttFile, "w") as f:
        f.write(attfileheader)

        # loop over all OD pairs
        for odKey in sorted(odDict):
            pathSeqIndexInsideOD = 1
            # loop over all trips of the current OD pair
            for tripAttr in odDict[odKey]:
                isLastTripOfTour = tripAttr[0]  # true, if the trip is the last trip of its tour

                # one path sequence item for each trip origin
                # OrigZoneNo of path sequence, DestZoneNo of path sequence, index of path sequence (per OD pair), index of path sequence item, OrigZoneNo of trip, DSegCode of the trip is PostDSegCode of the item, tour identifier PersonNo, ScheduleNo and TourNo
                f.write('1;%d;%d;%d;%d;%d;%s;1;%d;%d;%d\n' % (
                tripAttr[1], tripAttr[2], pathSeqIndexInsideOD, tripAttr[3], tripAttr[4], tripAttr[6], tripAttr[7],
                tripAttr[8], tripAttr[9]))

                if isLastTripOfTour:
                    # extra line for last path seq item = destination of last trip
                    # ..., DestZoneNo of trip, "", ...
                    f.write('1;%d;%d;%d;%d;%d;%s;1;%d;%d;%d\n' % (
                    tripAttr[1], tripAttr[2], pathSeqIndexInsideOD, tripAttr[3] + 1, tripAttr[5], "", tripAttr[7],
                    tripAttr[8], tripAttr[9]))

                    pathSeqIndexInsideOD += 1  # after last trip of the tour, the next tour resp. path sequence starts

    # import path sequence attribute file to Visum
    Visum.IO.ImportPathSequences(pathSeqItemsAttFile, 1)

if __name__ == "__main__":
    # start logging engine
    Visum = globals()['Visum']
    my_abm = abm.ABM(Visum=Visum, model_dir=r'.\\')
    run(Visum)
