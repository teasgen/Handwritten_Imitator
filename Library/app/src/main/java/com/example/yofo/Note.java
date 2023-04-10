package com.example.yofo;

public class Note {
    private int id;
    private String title;
    private String imgPath;

    Note(String imgPath, String title) {
        this.imgPath = imgPath;
        this.title = title;
    }

    Note(int id, String imgPath, String title) {
        this.imgPath = imgPath;
        this.id = id;
        this.title = title;
    }

    public String getImgPath() {
        return imgPath;
    }

    public int getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public void setImgPath(String imgPath) {
        this.imgPath = imgPath;
    }

    public void setId(int id) {
        this.id = id;
    }

    public void setTitle(String title) {
        this.title = title;
    }
}
