package com.example.yofo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.InputStream;

public class GenerationPage extends AppCompatActivity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.generated_result);

        InputStream inputStream = WriteTextPage.getInputStream();
        Bitmap gotImage = BitmapFactory.decodeStream(inputStream);
        ImageView imageView = findViewById(R.id.generated);
        imageView.setImageBitmap(gotImage);
    }
}
