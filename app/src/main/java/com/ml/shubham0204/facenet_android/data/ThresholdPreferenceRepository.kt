package com.ml.shubham0204.facenet_android.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.koin.core.annotation.Single

@Single
class ThresholdPreferenceRepository(context: Context) {
    private val prefs = context.getSharedPreferences("face_prefs", Context.MODE_PRIVATE)
    private val key = "cosine_threshold"
    private val default = 0.7f

    // In-memory cache
    @Volatile
    private var cachedThreshold: Float = prefs.getFloat(key, default)

    private val _threshold = MutableStateFlow(readThreshold())
    val threshold: StateFlow<Float> = _threshold

    private fun readThreshold(): Float {
        return prefs.getFloat(key, default)
    }

    fun getThreshold(): Float = threshold.value

    fun setThreshold(value: Float) {
        cachedThreshold = value
        prefs.edit().putFloat(key, value).apply()
        _threshold.value = value
    }
}