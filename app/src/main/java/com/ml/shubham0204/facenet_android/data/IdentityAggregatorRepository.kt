package com.ml.shubham0204.facenet_android.data

import android.content.Context
import android.util.Log
import androidx.collection.IntList
import androidx.collection.MutableIntList
import androidx.collection.MutableLongList
import org.koin.core.annotation.Single

@Single
class IdentityAggregatorRepository() {
    private val hashMap = HashMap<String, MutableLongList>()
    fun faceDetected(tracker_id:Int,person_id:Long):Long{
        Log.d("faceDetected", "faceDetected: This Function called")
        val list = hashMap[tracker_id.toString()]
//        hashMap[tracker_id.toString()]
        if (list != null) {
            if(list.size>=5){
                Log.d("faceDetected", "faceDetected: This Function called 2")
                val findMajority = findMajority(list)
                list.clear()
                hashMap.remove(tracker_id.toString())
                return findMajority
            }
            list.add(person_id)
            hashMap[tracker_id.toString()] = list
        }
        else{
            if (hashMap.size>2){
                Log.d("faceDetected", "faceDetected: Hash map called")
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
        if (max>3){
            Log.d("faceDetected", "faceDetected: Max Greater than five")
            return maxperson
        }
        else{
            Log.d("faceDetected", "faceDetected: MAx Value $max")
            return -1
        }


    }
}