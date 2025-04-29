package com.ml.shubham0204.facenet_android

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.ml.shubham0204.facenet_android.presentation.screens.add_face.AddFaceScreen
import com.ml.shubham0204.facenet_android.presentation.screens.detect_screen.DetectScreen
import com.ml.shubham0204.facenet_android.presentation.screens.face_list.FaceListScreen
import com.ml.shubham0204.facenet_android.util.BatchAddPersonImages
import com.ml.shubham0204.facenet_android.util.BatchedFileLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {

    private lateinit var folderPickerLauncher: ActivityResultLauncher<Intent>


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        BatchedFileLogger.init(applicationContext)
        enableEdgeToEdge()

        // Register the folder picker launcher
        folderPickerLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == RESULT_OK) {
                val folderUri = result.data?.data
                if (folderUri != null) {
                    // Persist permission if needed
                    contentResolver.takePersistableUriPermission(
                        folderUri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                    // Call your batch add logic here
                    lifecycleScope.launch {
                        BatchAddPersonImages.batchAddPersonsFromUserFolder(
                            context = applicationContext,
                            folderUri = folderUri
                        )
                    }
                }
            }
        }

        setContent {
            val navHostController = rememberNavController()
            NavHost(
                navController = navHostController,
                startDestination = "detect",
                enterTransition = { fadeIn() },
                exitTransition = { fadeOut() }
            ) {
                composable("add-face") { AddFaceScreen { navHostController.navigateUp() } }
                composable("detect") { DetectScreen { navHostController.navigate("face-list") } }
                composable("face-list") {
                    FaceListScreen(
                        onNavigateBack = { navHostController.navigateUp() },
                        onAddFaceClick = { navHostController.navigate("add-face") },
                        onExportLogClick = { exportLogFile() },
                        onClearLogClick = { clearLogFile() },
                        onBatchAddClick = { batchAddPerson() }
                    )
                }
            }
        }
    }

    private fun exportLogFile() {
        val logFile = File(filesDir, "app_log.txt")
        if (!logFile.exists()) {
            // Optionally show a toast/snackbar: "No log file to export"
            return
        }
        val uri: Uri = FileProvider.getUriForFile(
            this,
            "${packageName}.fileprovider",
            logFile
        )
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Export Log File"))
    }

    private fun clearLogFile() {
        val logFile = File(filesDir, "app_log.txt")
        if (logFile.exists()) {
            logFile.delete()
        }
    }

    private fun batchAddPerson() {
//        lifecycleScope.launch {
//            BatchAddPersonImages.batchAddPersonsFromAssets(
//                context = applicationContext,
//                assetFolder = "persons_spintly"
//            )
//        }

        // Launch the folder picker instead of using assets
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
        folderPickerLauncher.launch(intent)
    }


    override fun onDestroy() {
        BatchedFileLogger.shutdown()
        super.onDestroy()
    }
}
