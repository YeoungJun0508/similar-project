### 얼굴 감지 추출

출처:https://github.com/ageitgey/face_recognition

```
import os
import face_recognition

data_path='/content/drive/MyDrive/2024_공부/OML/img/pre_MAN'
image_path = os.listdir(data_path)

for i in image_path:
  image_files = os.path.join(data_path, i)
  image = face_recognition.load_image_file(image_files)
  face_locations = face_recognition.face_locations(image)
  print("I found {} face(s) in this photograph.".format(len(face_locations)))

  for face_location in face_locations:

      # Print the location of each face in this image
      top, right, bottom, left = face_location
      print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

      # You can access the actual face itself like this:
      face_image = image[top:bottom, left:right]
      pil_image = Image.fromarray(face_image)
      # pil_image.show()
      display(pil_image)
```

![image](https://github.com/YeoungJun0508/similar-project/assets/145903037/9ae2f68d-11df-4881-a21c-5c960d23ed81)

