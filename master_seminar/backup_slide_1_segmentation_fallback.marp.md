# Back-Up Slide 1: Segmentation Fallback Strategy

```text
            Video Input
                 |
      Primary: PySceneDetect
        (Threshold = 27)
                 |
        scene_count > 0 ?
            /         \
          Yes          No
           |            |
 Rapid cuts captured    Fallback: Fixed Temporal Windows
           |            Triggered if scene_count = 0
           |            (e.g., strict 5-second segments)
            \          /
             \        /
   Continuous narrative unit extraction
                  |
         Units passed to LLM reasoning
```

**Primary:** PySceneDetect (Threshold = 27), captures rapid cuts.

**Fallback:** Fixed Temporal Windows, triggered if scene_count = 0 (e.g., strict 5-second segments).

**Goal:** Continuous narrative unit extraction.

**What to say (Script):**
"When PySceneDetect fails due to hard cuts or artifacts, we don't just discard the video. Our fallback strategy, which we are implementing now, forces a fixed temporal window, for instance 5 seconds. This guarantees we always extract narrative units for the LLM to process."
