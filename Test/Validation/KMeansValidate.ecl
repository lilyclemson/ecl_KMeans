/**
The use of this code is to validate ECL KMeans Clustering Algorithm against the standard
public dataset Iris[1] with known results. As part of the performance evaluation,
the result of KMeans clustering analytics is compared to the result generated by a widely
used open source implementation SKlearn.cluster.KMeans [2].
The python code to generate the result of the SKlearn.cluster.KMeans can be
found via the directory 'Test\Validation\Python\sklean_KMeans.py'.
* Reference
* [1] Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository
      [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
      School of Information and Computer Science
* [2] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
*/

IMPORT ML_Core;
IMPORT ML_Core.Types;
IMPORT Cluster;
IMPORT Test;

//Data Preperation
//Load Iris Dataset
DSIris := Test.Datasets.DSIris.ds;
//Add ID to each record
ML_Core.AppendSeqId(DSIris, id, DSIrisWId);
//Transform the raw data into  Machine Learning Dataframe:
//ML_Core.Types.NumericField
ML_Core.ToField(DSIrisWId, DSIrisWIdWi);
//Define d01: data points
d01 := DSIrisWIdWi(number < 5); //Filter the attribute not used for clustering
//Define d02: initial centroids
//Three centroids are initialized with 0, 1, 2 as its id respectively
ids := [1,51,101];
d02 := PROJECT(d01(id IN ids),TRANSFORM(Types.NumericField,
                                        SELF.id := MAP( LEFT.id = 1   => 0,
                                                        LEFT.id = 51  => 1, 2),
                                        SELF := LEFT));
//Set up the parameters
max_iteratons := 30;
tolerance := 0.0;
//Train KMeans model with data points d01 and centroids d02
KMeansRst := Cluster.KMeans(max_iteratons, tolerance).fit(d01, d02);
//Below are the results:
//Coordinates of cluster centers
Centroids := KMeansRst.centroids;
//Number of iterations run
Total_Iterations := KMeansRst.tol_iters;
//Labels of each point
Labels := KMeansRst.labels;

//
//Validate the results against the results of sklearn.Cluster.KMeans
ML_Core.ToField(Test.Datasets.DSIris.sklearn_rst, sklearn_rst);
//Validation 1: Coordinates of each cluster center
Compare_Coordinate := JOIN(sklearn_rst,centroids,
                            LEFT.id = RIGHT.id AND LEFT.number = RIGHT.number,
                            TRANSFORM(Types.NumericField,
                            SELF.value := IF((DECIMAL10_8) LEFT.value = (DECIMAL10_8) RIGHT.value,
                                          1,
                                          0),
                            SELF := LEFT), LEFT OUTER);
IsSameCoordinate := IF(COUNT(Compare_Coordinate(value = 0)) = 0, TRUE, FALSE);
OUTPUT(IsSameCoordinate, NAMED('IsSameCoordinate'));
//Validation 2: Number of iteration run
sklearn_converge := Test.Datasets.DSIris.sklearn_converge;
IsSameNumberRun := IF( Total_Iterations[1].iter = sklearn_converge, TRUE, False);
OUTPUT(IsSameNumberRun, NAMED('IsSameNumberRun'));
//Validation 3: Label of each point
sklearn_labels:= Test.Datasets.DSIris.sklearn_alleg;
Compare_Label := JOIN(sklearn_labels, labels, LEFT.id = RIGHT.x,
                                      TRANSFORM({INTEGER id, BOOLEAN isSame},
                                      SELF.isSame := IF(LEFT.y = RIGHT.y, TRUE, FALSE),
                                      SELF := LEFT), LEFT OUTER);
IsSameLable := IF(COUNT(Compare_Label(isSame = FALSE)) = 0, TRUE, FALSE);
OUTPUT(IsSameLable , NAMED('IsSameLable'));
//IfValid if all the results are the same in two different implementations.
IfValidated := IF(IsSameNumberRun, IF(IsSameLable, IF(IsSameCoordinate, TRUE, FALSE), False), False);
OUTPUT(IfValidated, NAMED('Result_is_validated'));