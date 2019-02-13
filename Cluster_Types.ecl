IMPORT ML_Core.Types as Types;

/**
  * Type definition module for KMeans.
  */
EXPORT Cluster_Types := MODULE
  /**
    * Definition of the meaning of the indexes of the KMeans Model variables.
    * <p>Ind1 enumerates the first index, which
    * is used to determine which type of data is stored:<ul>
    * <li>Centers stores the list of centers of clusters.
    *             The second index is the centerID.
    *             The third index is the number field of the center.</li>
    * <li>samples stores the set of sample indexes (i.e. ids) associated
    *             with each centerId. The value is the Id of its closest center.
    * <li>Iterations  stores the iterations associated with each wi. It represents how
    *                 many iteration runs of each wi before it stops iterating.
    *                 It does not have following index.</li></ul>
    */
  EXPORT KMeans_Model := MODULE
    /**
      * Index 1 represents the category of data within the model.
      *
      * @value reserved = 1.   Reserved for future use.
      * @value centers = 2.    The set of tree nodes within the model.
      * @value samples = 3.    The particular record ids that are included in tree's sample .
      * @value iterations = 4. The iteration runs of each wi.
      */
    EXPORT Ind1 := MODULE
      EXPORT Types.t_index reserved := 1; // Reserved for future use
      EXPORT Types.t_index centers := 2;
      EXPORT Types.t_index samples := 3;
      EXPORT Types.t_index iterations := 4;
    END;
    /**
      * Ind2 enumerates the seoncd and third indexes of each center which is the
      * parent index. The parent index value is 2. It is used to store the id and
      * the field value of each center.
      *
      * @value id = 1.  The center identifier.
      * @value number = 2.  The field identifier.
      */
    EXPORT Ind2 := MODULE
      EXPORT Types.t_index id := 1;
      EXPORT Types.t_index number := 1;
    END;
    /**
      * Ind3 enumerates the indexes of each sample which is the
      * parent index. The parent index value is 3. It is used to store
      * the sampleID. The value is the Id of its closest center.
      *
      * @value id = 1.  The sample identifier.
      * @value
      */
    EXPORT Ind3 := MODULE
      EXPORT Types.t_index id := 1;
    END;
    EXPORT Centers_Indexes := ENUM(UNSIGNED2,  id=2, number=3);
    EXPORT Samples_Indexes := ENUM(UNSIGNED2,  id=2);
    /**
      * Labels format defines the distance space where each cluster defined by
      * a center and its closest samples.
      *
      * @value wi      The model identifier.
      * @value id      The sample identifier.
      * @value lable   The identifier of the closest center to the sample.
      */
    EXPORT Labels := RECORD
      Types.t_Work_Item wi;      // Model Identifier
      Types.t_RecordID  id;      // Sample Identifier
      Types.t_RecordID  label;   // Center Identifier
    END;
  END;//KMeans-Types
END; //Cluster_Types