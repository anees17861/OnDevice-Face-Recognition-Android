package com.ml.shubham0204.facenet_android.data

import android.util.Log
import com.ml.shubham0204.facenet_android.util.BatchedFileLogger
import org.koin.core.annotation.Single
import kotlin.math.sqrt

@Single
class ImagesVectorDB {

    private val imagesBox = ObjectBoxStore.store.boxFor(FaceImageRecord::class.java)

    fun addFaceImageRecord(record: FaceImageRecord) {
        imagesBox.put(record)
    }

    fun getNearestEmbeddingPersonName(embedding: FloatArray): FaceImageRecord? {
        /*
        Use maxResultCount to set the maximum number of objects to return by the ANN condition.
        Hint: it can also be used as the "ef" HNSW parameter to increase the search quality in combination
        with a query limit. For example, use maxResultCount of 100 with a Query limit of 10 to have 10 results
        that are of potentially better quality than just passing in 10 for maxResultCount
        (quality/performance tradeoff).
         */
        BatchedFileLogger.log("getNearestEmbeddingPersonName begin")
//        val topMatch = imagesBox
//            .query(FaceImageRecord_.faceEmbedding.nearestNeighbors(embedding, 5))
//            .build()
//            .findWithScores()
//            .maxByOrNull {
//                it.score
//                Log.d("getNearestEmbeddingPersonName", "score: ${it.score} ${it.get().personName}")
//            }  // highest cosine similarity (since vectors are normalized)
//            ?.get()  // extract the actual FaceImageRecord, or null if none
//            BatchedFileLogger.log("getNearestEmbeddingPersonName results: ${it.score} ${it.get().personName}")


        return imagesBox
            .query(FaceImageRecord_.faceEmbedding.nearestNeighbors(embedding, 10))
            .build()
            .findWithScores()
            .map {
                BatchedFileLogger.log("getNearestEmbeddingPersonName results: ${it.score} ${it.get().personName}")
                it.get()
            }
            .firstOrNull()
    }

    fun removeFaceRecordsWithPersonID(personID: Long) {
        imagesBox.removeByIds(
            imagesBox.query(FaceImageRecord_.personID.equal(personID)).build().findIds().toList()
        )
    }
}
