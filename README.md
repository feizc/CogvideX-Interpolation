## CogvideX-Interpolation: Keyframe Interpolation with CogvideoX

CogVideoX-Interpolation is a modified pipeline based on the CogVideoX structure, designed to provide more flexibility in keyframe interpolation generation. 

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input frame 1</td>
        <td>Input frame 2</td>
        <td>Text</td>
        <td>Generated video</td>
    </tr>
  	<tr>
	  <td>
	    <img src=cases/5.jpg width="250">
	  </td>
	  <td>
	    <img src=cases/55.jpg width="250">
	  </td>
      <td>
	    A group of people dance in the street at night, forming a circle and moving in unison, exuding a sense of community and joy. A woman in an orange jacket is actively engaged in the dance, smiling broadly. The atmosphere is vibrant and festive, with other individuals participating in the dance, contributing to the sense of community and joy. 
	  </td>
	  <td>
	    <img src=cases/gen.mp3 width="250">
	  </td>
  	</tr>
  	<tr>
	  <td>
	    <img src=cases/6.jpg width="250">
	  </td>
	  <td>
	    <img src=cases/66.jpg width="250">
	  </td>
      <td>
	    A man in a white suit stands on a stage, passionately preaching to an audience. The stage is decorated with vases with yellow flowers and a red carpet, creating a formal and engaging atmosphere. The audience is seated and attentive, listening to the speaker. 
	  </td>
	  <td>
	    <img src=cases/gen.mp3 width="250">
	  </td>
  	</tr>
  <tr>
	  <td>
	    <img src=cases/8.jpg width="250">
	  </td>
	  <td>
	    <img src=cases/88.jpg width="250">
	  </td>
      <td>
	    A man runs on a red track, marked with white lane lines, wearing a dark blue and white athletic outfit and yellow shoes. He is in a competitive environment, with blurred figures in the background. 
	  </td>
	  <td>
	    <img src=cases/gen.mp3 width="250">
	  </td>
  	</tr>
</table >


## Quick Start
### 1. Setup repository and environment
```
pip install -r requirement.txt
```


### 2. Download checkpoint
Download the finetuned [checkpoint](huggingface), and put it under `checkpoints/`. 

### 3. Launch the inference script!
The example input keyframe pairs are in `infer` folder. 


## Light-weight finetuing


## Knowledge 








