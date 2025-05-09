package com.ml.shubham0204.facenet_android

import android.app.Application
import com.ml.shubham0204.facenet_android.data.ObjectBoxStore
import com.ml.shubham0204.facenet_android.di.AppModule
import org.koin.android.ext.koin.androidContext
import org.koin.core.context.startKoin
import org.koin.ksp.generated.module
import org.opencv.android.OpenCVLoader
import android.util.Log

class MainApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        startKoin {
            androidContext(this@MainApplication)
            modules(AppModule().module)
        }
        ObjectBoxStore.init(this)
        // Initialize OpenCV
        if (OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "OpenCV initialization successful")
        } else {
            Log.e("OpenCV", "OpenCV initialization failed")
        }
    }
}
