package com.example.yofo;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class WriteTextPage extends AppCompatActivity {
    private final String url="http://192.168.1.40:5000";
    private static InputStream inputStream;

    public static InputStream getInputStream() {
        return inputStream;
    }

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.about_book);
        Objects.requireNonNull(getSupportActionBar()).hide();
        ImageView imageView = findViewById(R.id.imageAbout);

        NotesDataset mDatabase = NotesDataset.getInstance(this);
        Intent prevIntent = getIntent();
        Note note = mDatabase.listNotes().get(Integer.parseInt(prevIntent.getStringExtra("position")));

        Uri imgUri = Uri.parse("file://" + note.getImgPath());
        Bitmap bitmap = null;
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imgUri);
        } catch (IOException e) {
            e.printStackTrace();
        }

        imageView.setImageBitmap(bitmap);

        Button buttonGenerate = findViewById(R.id.buttonGenerateText);
        buttonGenerate.setOnClickListener(v -> {
            new Thread(() -> {
                File file = new File(note.getImgPath());
                EditText editText = findViewById(R.id.addText);
                String text = String.valueOf(editText.getText());
                RequestBody requestBody = new MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart("text", text)
                        .addFormDataPart("font", file.getName(), RequestBody.create(MediaType.parse("image/jpeg"), file))
                        .build();

                Request request = new Request.Builder()
                        .url(url + "/upload")
                        .post(requestBody)
                        .build();

                OkHttpClient client = new OkHttpClient();
                Response response = null;
                try {
                    response = client.newCall(request).execute();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                byte[] imageData = new byte[0];
                try {
                    assert response != null;
                    imageData = response.body().bytes();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                inputStream = new ByteArrayInputStream(imageData);
                Intent intent = new Intent(this, GenerationPage.class);
                startActivity(intent);
            }).start();
        });
    }
}