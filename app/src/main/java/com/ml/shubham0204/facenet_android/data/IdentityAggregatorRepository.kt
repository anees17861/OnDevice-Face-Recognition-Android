package com.ml.shubham0204.facenet_android.data

import android.content.Context
import android.util.Log
import androidx.collection.IntList
import androidx.collection.MutableIntList
import androidx.collection.MutableLongList
import org.koin.core.annotation.Single
import com.ml.shubham0204.facenet_android.util.BatchedFileLogger



@Single
class IdentityAggregatorRepository() {
    private val hashMap = HashMap<String, MutableLongList>()
    fun faceDetected(tracker_id:Int,person_id:Long):Long{
        BatchedFileLogger.log("IdentityAggregatorRepository, facesAggregated: Initialized")
        val list = hashMap[tracker_id.toString()]
//        hashMap[tracker_id.toString()]
        if (list != null) {
            list.add(person_id)
            if(list.size>=3){
                val findMajority = findMajority(list)
                list.clear()
                hashMap.remove(tracker_id.toString())
                return findMajority
            }
            hashMap[tracker_id.toString()] = list
        }
        else{
            if (hashMap.size>2){
                hashMap.remove(hashMap.keys.minOf { it })
            }
            val newList = MutableLongList()
            newList.add(person_id)
            hashMap[tracker_id.toString()] = newList
        }
        return -1
//        array[]
    }
    private fun findMajority(list: MutableLongList):Long{
        val map = HashMap<Long,Int>()
        var max = 0
        var maxperson = -1L
        list.forEach {
            if(map.containsKey(it)){
                map[it] = map[it]!!+1
            }
            else {
                map[it] = 1
            }
        }
        map.forEach {
            if(it.value>max){
                max = it.value
                maxperson = it.key

            }
        }
        if (max>2){
            BatchedFileLogger.log("IdentityAggregatorRepository, facesAggregated: $max")
            return maxperson
        }
        else{
            return -1
        }


    }
    fun clearFace(tracker_id: Int){
        hashMap.remove(tracker_id.toString())

    }
}