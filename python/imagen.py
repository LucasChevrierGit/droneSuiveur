import cv2

def extract_frames(video_path, output_dir):
    # Load video
    video = cv2.VideoCapture(video_path)
    
    # Check video is opened
    if not video.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        return
    
    # Count for images
    frame_count = 1
    
    # Read the frames until the end of the video
    while True:
        # Read a single frame
        ret, frame = video.read()
        
        # Check if it is read properly
        if not ret:
            break
        
        # Output path for the image to be saved to
        output_path = f"{output_dir}/{frame_count}.jpg"
        
        # Save as .jpg
        cv2.imwrite(output_path, frame)
        
        # show the result
        print(f"Frame {frame_count} extraite et sauvegardée.")
        
        # Increment the count
        frame_count += 1
    
    # Close the video
    video.release()

# Main call for the above function
video_path = "../AS-One/data/sample_videos/walk_test.mp4"
output_dir = "../dataset/images/val"
extract_frames(video_path, output_dir)