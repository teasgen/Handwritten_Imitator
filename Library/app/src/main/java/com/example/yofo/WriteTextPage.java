package com.example.yofo;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

public class WriteTextPage extends AppCompatActivity {
    String PDFormat;
    private final Object lock = new Object();
    EditText alertInput;

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

        ImageButton buttonGenerate = findViewById(R.id.buttonGenerateText);
        buttonGenerate.setOnClickListener(v -> {
            showAlertDialog();
            new Thread(() -> {
                synchronized (lock) {
                    try {
                        lock.wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    File file = new File(note.getImgPath());
                    EditText editText = findViewById(R.id.addText);
                    String text = String.valueOf(editText.getText());
                    Intent intent = new Intent(this, GenerationPage.class);
                    intent.putExtra("text", text);
                    intent.putExtra("file", file);
                    intent.putExtra("format", PDFormat);
                    intent.putExtra("name", alertInput.getText().toString());
                    startActivity(intent);
                }
            }).start();
        });
    }

    private void setPDFormat(String format) {
        synchronized (lock) {
            PDFormat = format;
            lock.notifyAll();
        }
    }

    private void showAlertDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        synchronized (lock) {
            alertInput = new EditText(this);
            alertInput.setHint("Write file name");
            builder.setTitle("Choose the PDF format")
                    .setView(alertInput)
                    .setNegativeButton("A4", (dialog, which) -> {
                        setPDFormat("A4");
                        dialog.dismiss();
                    })
                    .setPositiveButton("A5", (dialog, which) -> {
                        setPDFormat("A5");
                        dialog.dismiss();
                    }).create();
            builder.show();
        }
    }
}
