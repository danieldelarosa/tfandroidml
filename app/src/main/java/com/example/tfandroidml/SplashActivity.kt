package com.example.tfandroidml

import android.content.Intent
import android.os.Handler
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Looper

class SplashActivity:AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        Handler(Looper.getMainLooper()).postDelayed({
            val intent = Intent(this@SplashActivity,MainActivity::class.java)
            startActivity(intent)
            finish()
        },2000)
    }
}