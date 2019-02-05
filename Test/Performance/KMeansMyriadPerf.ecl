/**
The use of this code is to test the Myriad interface of ECL KMeans Clustering Algorithm
using the standard public dataset Iris[1].
* Reference
* [1] Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository
      [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
      School of Information and Computer Science
*/

IMPORT ML_Core;
IMPORT ML_Core.Types;
IMPORT Cluster;
IMPORT PBblas;
IMPORT Test;

//Data Preperation
//Load Iris Dataset
ds := Test.Datasets.DSIris.ds;
//Setup performance baseline
//Baseline-single wi
centroidsID := [1,51,101];
//Set up the parameters
max_iteratons := 30;
tolerance := 0.0;
ML_Core.AppendSeqId(ds, id, base_ids);
ML_Core.ToField(base_ids, base_idsWi);
base_d01 := base_idsWi(number < 5);
base_d02 := base_d01(id IN centroidsID);
base_Model := Cluster.KMeans(max_iteratons, tolerance).fit(base_d01, base_d02);
base_Centroids := base_Model.Centroids;
Base_iters := base_Model.tol_iters;
base_labels := base_Model.labels;

//Create myriad dataset: multi-wi [1..5]
myriad(INTEGER w) := PROJECT(base_d01, TRANSFORM(Types.NumericField,
                                                  SELF.wi := w,
                                                  SELF := LEFT));
ds1 := myriad(1);
ds2 := myriad(2);
ds3 := myriad(3);
ds4 := myriad(4);
ds5 := myriad(5);

//Combine the above datasets into one dataset d01
//d01: the combined dataset with multi-wi
d01  := ds1 + ds2 + ds3 + ds4 + ds5;
//d02 is the centroids of each model/wi corresponding the wi of d01
d02 := d01(id IN centroidsID);

//Run KMeans on d01 and d02
Myriad_Model :=Cluster.KMeans(max_iteratons, tolerance).fit(d01, d02);
//Coordinates of cluster centers
Centroids := Myriad_Model.Centroids;
//Number of iterations run
iters := Myriad_Model.tol_iters;
//Labels of each point
labels := Myriad_Model.labels;

//Compare myriad-interface with baseline
Compare_Coordinate := JOIN(Centroids, base_Centroids,
                        LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number,
                                TRANSFORM(Types.NumericField,
                                          SELF.value :=  (DECIMAL10_8)LEFT.value - (DECIMAL10_8)RIGHT.value;
                                          SELF := LEFT),
                      LOOKUP);
IsSameCoordinate := IF(COUNT(Compare_Coordinate(value <> 0)) = 0, TRUE, FALSE);
OUTPUT(IsSameCoordinate, NAMED('IsSameCoordinate'));
Compare_runs := JOIN(base_iters, iters, LEFT.iter <> RIGHT.iter, TRANSFORM(LEFT),ALL);
IsSameNumberRun := IF(COUNT(Compare_runs) <> 0, FALSE , TRUE);
OUTPUT(IsSameNumberRun , NAMED('IsSameNumberRun'));
Compare_Label := JOIN(labels, base_labels,
                        LEFT.x = RIGHT.x AND LEFT.y <> RIGHT.y,
                        LOOKUP);
IsSameLable :=  IF(COUNT(Compare_Label) = 0, TRUE, FALSE);
OUTPUT(IsSameLable , NAMED('IsSameLable'));
IsMyriadWork := IF(IsSameNumberRun, IF(IsSameLable, IF(IsSameCoordinate, TRUE, FALSE), False), False);
OUTPUT(IsMyriadWork, NAMED('IsMyriadWork'));