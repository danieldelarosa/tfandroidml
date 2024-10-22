package com.example.tfandroidml

import android.content.Intent
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.DataType
import com.example.tfandroidml.ml.Classification


class MainActivity : AppCompatActivity() {
    lateinit var TakePhoto:Button
    lateinit var ChoosePhoto:Button
    lateinit var Result:TextView
    lateinit var Image:ImageView
    lateinit var Predict:Button
    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        TakePhoto = findViewById(R.id.button)
        ChoosePhoto = findViewById(R.id.button2)
        Result = findViewById(R.id.textView)
        Image = findViewById(R.id.imageView)
        Predict = findViewById(R.id.button3)

        val labels=application.assets.open("labels.txt").bufferedReader().readLines()

        val imageProcessor=ImageProcessor.Builder()
            .add(ResizeOp(224,224,ResizeOp.ResizeMethod.BILINEAR))
            .build()

        TakePhoto.setOnClickListener{
            val intent: Intent=Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent,2)
        }

        ChoosePhoto.setOnClickListener{
            var intent: Intent=Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(Intent.createChooser(intent,"Selecciona una imagen"),1)
        }

        Predict.setOnClickListener{
            if (!::bitmap.isInitialized){
                Toast.makeText(this,"Por favor seleccione una imagen",Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            var tensorImage = TensorImage(DataType.UINT8)
            tensorImage.load(bitmap)
            tensorImage=imageProcessor.process(tensorImage)

            val model = Classification.newInstance(this)

            val inputFeature0=TensorBuffer.createFixedSize(intArrayOf(1,224,224,3),DataType.UINT8)
            inputFeature0.loadBuffer(tensorImage.buffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0=outputs.outputFeature0AsTensorBuffer.floatArray

            var maxIndx=0
            outputFeature0.forEachIndexed{ index,fl ->
                if(outputFeature0[maxIndx]<fl){
                    maxIndx=index
                }
            }
            Result.setText(labels[maxIndx])
            model.close()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(requestCode == 1){
            var uri = data?.data;
            bitmap=MediaStore.Images.Media.getBitmap(this.contentResolver,uri)
            Image.setImageBitmap(bitmap)
        }else if (requestCode == 2){
            bitmap = data?.extras?.get("data") as Bitmap
            Image.setImageBitmap(bitmap)
        }
    }
}