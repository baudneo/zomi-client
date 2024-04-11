# Strategy

There are a few strategies for matching. A matching object will need to be detected and then pass through filtering 
(confidence, name of object, location/area of object, etc.)


## Fast
These strategies are fast but may not give you all the information you need. You will get 1 notification ASAP.

### `fast`
*Will only ever return one object.*

This is the fastest strategy but, you may miss some information. As soon as a detected object makes it through filtering, no more processing occurs. The image and object are returned. 

### `first`
The first image with any detected objects will be returned. This is 2nd fastest as it lets the processing continue for all objects in the image and can return multiple objects. 

## Slow
These strategies are slow due to iterating the live stream while an event is happening. Slow strategies offer the 
ability for 'ongoing' notifications while processing; receive a notification about each unique filtered object.
## `most`
iterate the whole event and return the image with the most detected objects. This is a slow strategy as it iterates the entire event and all detected objects.

## `unique`
iterate the whole event and return the image with the most unique (person, person, person = 1 object, would lose to: 1 person, 1 dog = 2 objects) detected objects. This is a slow strategy as it iterates the entire event and all detected objects.

## `conf`
iterate the whole event and return the image with the highest sum of confidences (person 50%, cat 50%, dog 30% = 130, would lose to: person 80%, dog 80% = 160). This is a slow strategy as it iterates the entire event and all detected objects.

## `target`
target takes a label and will iterate the whole event to return the image with the most labels of the target label. Any ties will use confidence summition to break the tie. This is a slow strategy as it iterates the entire event and all detected objects.