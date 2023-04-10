package com.example.yofo;

public class Image {
    private String name;
    private int imageRes;

    public Image(String name, int image) {
        this.name = name;
        this.imageRes = image;
    }

    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getImageResource() {
        return this.imageRes;
    }

    public void setImageResource(int imageRes) {
        this.imageRes = imageRes;
    }
}
