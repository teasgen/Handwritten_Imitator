package com.example.yofo;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Filter;
import android.widget.Filterable;
import android.widget.ImageView;
import android.widget.PopupMenu;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.io.IOException;
import java.util.ArrayList;


public class CustomRecyclerAdapter extends
        RecyclerView.Adapter<CustomRecyclerAdapter.ViewHolder> implements Filterable {

    private final ItemClickListener itemClickListener;
    private Context context;
    private ArrayList<Note> listNotes;
    private ArrayList<Note> mArrayList;
    private NotesDataset mDatabase;

    CustomRecyclerAdapter(Context context, ArrayList<Note> listNotes, ItemClickListener itemClickListener) {
        this.context = context;
        this.listNotes = listNotes;
        this.mArrayList = listNotes;
        mDatabase = new NotesDataset(context);
        this.itemClickListener = itemClickListener;
    }
    @SuppressLint("SetTextI18n")
    public void onBindViewHolder(CustomRecyclerAdapter.ViewHolder holder, int
            position) {
        final Note note = listNotes.get(position);
        holder.label.setText(note.getTitle());
        Uri imgUri = Uri.parse("file://" + note.getImgPath());
        holder.image.setImageURI(null);
        try {
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(context.getContentResolver(), imgUri);
            holder.image.setImageBitmap(bitmap);
        } catch (IOException e) {
            e.printStackTrace();
        }
        holder.item_menu.setOnClickListener(v -> {
            PopupMenu popup = new PopupMenu(v.getContext(), holder.item_menu);
            popup.inflate(R.menu.item_menu);
            popup.show();
            popup.setOnMenuItemClickListener(item -> {
                if (item.getItemId() == R.id.change_book) {
                    Intent intent = new Intent(this.context,MakingNote.class);
                    intent.putExtra("position", String.valueOf(position));
                    context.startActivity(intent);
                    return true;
                } else if (item.getItemId() == R.id.delete) {
                    mDatabase.deleteNote(mDatabase.listNotes().get(position).getId());
                    listNotes = mDatabase.listNotes();
                    mArrayList = listNotes;
                    notifyItemRemoved(position);
                }
                return false;
            });
        });
    }

    @Override
    public Filter getFilter() {
        return new Filter() {
            @Override
            protected FilterResults performFiltering(CharSequence charSequence) {
                String charString = charSequence.toString();
                if (charString.isEmpty()) {
                    listNotes = mArrayList;
                }
                else {
                    ArrayList<Note> filteredList = new ArrayList<>();
                    for (Note contacts : mArrayList) {
                        if (contacts.getTitle().toLowerCase().contains(charString)) {
                            filteredList.add(contacts);
                        }
                    }
                    listNotes = filteredList;
                }
                FilterResults filterResults = new FilterResults();
                filterResults.values = listNotes;
                return filterResults;
            }

            @SuppressLint("NotifyDataSetChanged")
            @Override
            protected void publishResults(CharSequence constraint, FilterResults results) {
                listNotes = (ArrayList<Note>) results.values;
                notifyDataSetChanged();
            }
        };
    }

    @NonNull
    @Override
    public CustomRecyclerAdapter.ViewHolder onCreateViewHolder(@NonNull
                                                                       ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item, parent, false);
        int height = Resources.getSystem().getDisplayMetrics().heightPixels;
        view.getLayoutParams().height = height / 5;
        return new ViewHolder(view, itemClickListener);
    }
    @Override
    public int getItemCount() {
        return listNotes.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        ImageView image;
        TextView label, item_menu;
        ViewHolder(View view, ItemClickListener itemClickListener){
            super(view);
            image = view.findViewById(R.id.imageView);
            label = view.findViewById(R.id.titleBook);
            item_menu = view.findViewById(R.id.dots);

            view.setOnClickListener(view1 -> {
                if (itemClickListener != null) {
                    int pos = getBindingAdapterPosition();

                    if (pos != RecyclerView.NO_POSITION) {
                        itemClickListener.onItemClick(pos);
                    }
                }
            });
        }
    }
}

