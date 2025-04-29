package com.ml.shubham0204.facenet_android.util

import android.content.Context
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

object BatchedFileLogger {
    private const val LOG_FILE_NAME = "app_log.txt"
    private const val BATCH_SIZE = 10                // Number of logs before flush
    private const val FLUSH_INTERVAL_MS = 2000L      // Flush every 2 seconds if not already flushed

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.getDefault())
    private val logBuffer = mutableListOf<String>()
    private val bufferLock = Any()
    private var flushJob: Job? = null
    private val isInitialized = AtomicBoolean(false)
    private lateinit var appContext: Context

    private val loggerScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    fun init(context: Context) {
        if (isInitialized.compareAndSet(false, true)) {
            appContext = context.applicationContext
            startPeriodicFlush()
        }
    }

    fun log(message: String) {
        val timestamp = dateFormat.format(Date())
        val logMessage = "$timestamp: $message\n"
        synchronized(bufferLock) {
            logBuffer.add(logMessage)
            if (logBuffer.size >= BATCH_SIZE) {
                flushBuffer()
            }
        }
    }

    private fun startPeriodicFlush() {
        flushJob = loggerScope.launch {
            while (isActive) {
                delay(FLUSH_INTERVAL_MS)
                synchronized(bufferLock) {
                    if (logBuffer.isNotEmpty()) {
                        flushBuffer()
                    }
                }
            }
        }
    }

    private fun flushBuffer() {
        if (!::appContext.isInitialized || logBuffer.isEmpty()) return
        val logsToWrite = logBuffer.joinToString(separator = "\n")
        logBuffer.clear()
        loggerScope.launch {
            try {
                val logFile = File(appContext.filesDir, LOG_FILE_NAME)
                FileOutputStream(logFile, true).use { fos ->
                    fos.write(logsToWrite.toByteArray())
                }
            } catch (e: Exception) {
                // Optionally handle/log error writing to file
                e.printStackTrace()
            }
        }
    }

    fun shutdown() {
        flushJob?.cancel()
        synchronized(bufferLock) {
            if (logBuffer.isNotEmpty()) {
                flushBuffer()
            }
        }
    }
}