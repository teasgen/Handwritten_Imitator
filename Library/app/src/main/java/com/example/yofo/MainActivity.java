package com.example.yofo;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.Toast;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements ItemClickListener{
    RecyclerView.LayoutManager RecyclerViewLayoutManager;
    NotesDataset mDatabase;
    CustomRecyclerAdapter adapter = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.activity_main);
        //Objects.requireNonNull(getSupportActionBar()).setDisplayShowTitleEnabled(false);
        RecyclerView recyclerView = findViewById(R.id.review1);
        RecyclerViewLayoutManager
                = new LinearLayoutManager(
                getApplicationContext());
        recyclerView.setLayoutManager(
                RecyclerViewLayoutManager);
        LinearLayoutManager HorizontalLayout = new LinearLayoutManager(
                MainActivity.this,
                LinearLayoutManager.VERTICAL,
                false);
        recyclerView.setLayoutManager(HorizontalLayout);

        mDatabase = NotesDataset.getInstance(this);
        ArrayList<Note> allNotes = mDatabase.listNotes();

        if (allNotes.size() > 0) {
            recyclerView.setVisibility(View.VISIBLE);
            adapter = new CustomRecyclerAdapter(this, allNotes, this);
            recyclerView.setAdapter(adapter);
        } else {
            recyclerView.setVisibility(View.GONE);
            Toast.makeText(this, "There is no note in the database. Start adding now", Toast.LENGTH_LONG).show();
        }

        ImageButton pageUploading = findViewById(R.id.makeNote);
        pageUploading.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this,
                    MakingNote.class);
            intent.putExtra("position", "-1");
            startActivity(intent);
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mDatabase != null) {
            mDatabase.close();
        }
    }

    @Override
    public void onItemClick(int position) {
        Intent intent = new Intent(MainActivity.this, WriteTextPage.class);
        intent.putExtra("position", String.valueOf(position));
        startActivity(intent);
    }
}