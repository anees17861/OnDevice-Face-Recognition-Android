package com.ml.shubham0204.facenet_android.util

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import com.ml.shubham0204.facenet_android.domain.ImageVectorUseCase
import com.ml.shubham0204.facenet_android.domain.PersonUseCase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.koin.java.KoinJavaComponent.inject

object BatchAddPersonImages {

    private val personUseCase: PersonUseCase by inject(PersonUseCase::class.java)
    private val imageVectorUseCase: ImageVectorUseCase by inject(ImageVectorUseCase::class.java)


    /**
     * Batch adds persons and their images from the assets folder.
     * Each image file name should be the person's name (e.g., "John_Doe.jpg").
     *
     * @param context The application context.
     * @param assetFolder The folder in assets containing the images (default: "persons").
     * @param imageExtensions Allowed image file extensions (default: jpg, jpeg, png).
     */
    suspend fun batchAddPersonsFromAssets(
        context: Context,
        assetFolder: String = "persons",
        imageExtensions: List<String> = listOf("jpg", "jpeg", "png")
    ) = withContext(Dispatchers.IO) {
        val assetManager = context.assets
        val imageFiles = assetManager.list(assetFolder)?.filter { file ->
            imageExtensions.any { ext -> file.endsWith(".$ext", ignoreCase = true) }
        } ?: emptyList()

        for (fileName in imageFiles) {
            // Extract person name from file name (remove extension, replace underscores with spaces)
            val personName = fileName.substringBeforeLast('.').replace('_', ' ')
            try {
                // Add person and get personID
                val personID = personUseCase.addPerson(personName, 1)
                // Prepare Uri for the asset file
                val assetFilePath = "$assetFolder/$fileName"
                val tempFile = createTempFileFromAsset(context, assetFilePath)
                val imageUri = Uri.fromFile(tempFile)
                // Add image to vector DB
                val result = imageVectorUseCase.addImage(personID, personName, imageUri)
                if (result.isSuccess) {
                    BatchedFileLogger.log("Successfully added $personName from $fileName")
                } else {
                    BatchedFileLogger.log("Failed to add $personName from $fileName: ${result.exceptionOrNull()?.message}")
                }
                // Clean up temp file
                tempFile.delete()
            } catch (e: Exception) {
                BatchedFileLogger.log("Error processing $fileName: ${e.message}")
            }
        }

        BatchedFileLogger.log("Batch add completed")
    }

    /**
     * Batch adds persons and their images from a user-selected folder (SAF).
     * Each image file name should be the person's name (e.g., "John_Doe.jpg").
     *
     * @param context The application context.
     * @param folderUri The SAF Uri of the folder selected by the user.
     * @param imageExtensions Allowed image file extensions (default: jpg, jpeg, png).
     */
    suspend fun batchAddPersonsFromUserFolder(
        context: Context,
        folderUri: Uri,
        imageExtensions: List<String> = listOf("jpg", "jpeg", "png")
    ) = withContext(Dispatchers.IO) {
        val contentResolver = context.contentResolver
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(
            folderUri,
            DocumentsContract.getTreeDocumentId(folderUri)
        )

        val projection = arrayOf(
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_MIME_TYPE
        )

        val cursor = contentResolver.query(childrenUri, projection, null, null, null)
        cursor?.use {
            while (it.moveToNext()) {
                val name = it.getString(0)
                val documentId = it.getString(1)
                val mimeType = it.getString(2)

                if (mimeType != null && mimeType.startsWith("image/") &&
                    imageExtensions.any { ext -> name.endsWith(".$ext", ignoreCase = true) }
                ) {
                    val personName = name.substringBeforeLast('.').replace('_', ' ')
                    try {
                        val personID = personUseCase.addPerson(personName, 1)
                        val imageUri = DocumentsContract.buildDocumentUriUsingTree(folderUri, documentId)
                        val result = imageVectorUseCase.addImage(personID, personName, imageUri)
                        if (result.isSuccess) {
                            BatchedFileLogger.log("Successfully added $personName from $name")
                        } else {
                            BatchedFileLogger.log("Failed to add $personName from $name: ${result.exceptionOrNull()?.message}")
                        }
                    } catch (e: Exception) {
                        BatchedFileLogger.log("Error processing $name: ${e.message}")
                    }
                }
            }
        }
        BatchedFileLogger.log("Batch add from user folder completed")
    }

    /**
     * Copies an asset file to a temporary file and returns the File.
     */
    private fun createTempFileFromAsset(context: Context, assetPath: String): java.io.File {
        val inputStream = context.assets.open(assetPath)
        val tempFile = java.io.File.createTempFile("person_asset_", null, context.cacheDir)
        tempFile.outputStream().use { output ->
            inputStream.copyTo(output)
        }
        inputStream.close()
        return tempFile
    }
}