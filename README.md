# Human-robot-Collaborative-desassembly


The collaboration between robots and humans (HRC) is becoming a staple of modern industry. Unlike conventional automation, collaborative robots are built to work alongside people and share common environments. This could increase productivity, flexibility and safety while still maintaining human expertise at a core of complex tasks.

This project demonstrates an intelligent collaborative disassembly system, which involves robotics technology, visual computing technology and human–machine interaction technology. The aim of the task is to facilitate a human operator with the aid of a collaborative robot in a disassembly operation, hence guaranteeing that safe and efficient collaboration ensue.

The collaborative robot, ABB YuMi IRB 14000 manufactured by ABB and 3D perception camera Intel RealSense D435 provided by Intel are used in the system.
The platform enables:
  1.Object and component detection
  2.Monitoring of operator actions
  3.Human and robot communicate using gestures
  4.Synchronization of operator and robot tasks

The project is organised in two major sections.

Here's a breakdown of the key areas we'll be focusing on:

  A.Modeling and Simulation:
    We started by designing the collaborative robotic cell. This involves simulating various scenarios to understand how the robot and humans will interact. We payed close attention to      safety zones and how humans and robots will work together.
    
  B.Vision and Interaction: 
    Next, we used computer vision to detect objects. We also tracked the disassembly process and used gesture recognition to enable seamless communication between humans and robots.

The ultimate aim is to showcase how collaborative robotics can significantly enhance industrial processes, specifically in areas like recycling, maintenance, and product disassembly.

[![Voir la vidéo](https://img.youtube.com/vi/9u7nuZ9OHc0/0.jpg)](https://www.youtube.com/watch?v=9u7nuZ9OHc0)

[![Voir la vidéo](https://img.youtube.com/vi/f7UhQudZ7B0/0.jpg)](https://www.youtube.com/watch?v=f7UhQudZ7B0)

[![Voir la vidéo](https://img.youtube.com/vi/zVySbl7mqOA/0.jpg)](https://www.youtube.com/watch?v=zVySbl7mqOA)

[![Voir la vidéo](https://img.youtube.com/vi/ELmnUNqyEkQ/0.jpg)](https://www.youtube.com/watch?v=ELmnUNqyEkQ)

[![Voir la vidéo](https://img.youtube.com/vi/SiB9le508Pc/0.jpg)](https://www.youtube.com/watch?v=SiB9le508Pc)


### Conclusion and Results

The results obtained in this project demonstrate the feasibility and potential of our **human–robot collaboration (HRC)** for assisted disassembly tasks. The developed system successfully computer vision, and human–machine interaction to support a human operator during the disassembly of an hydraulic cylinder prototype. Using the **ABB YuMi IRB 14000** and the **Intel RealSense D435**, the system was able to detect and track several components of the cylinder during the disassembly process.

The trained **YOLO** segmentation model allowed the detection of different cylinder parts in real time. Although some minor detection errors may occur during live operation, this does not significantly affect the workflow. A dedicated **verification protocol** was implemented to confirm the disassembly state and ensure the reliability of the process. This additional validation layer allows the system to maintain accurate tracking of the operation despite occasional detection inaccuracies.

An application was developed to manage the overall disassembly workflow. Based on the data received from the vision and interaction modules, the application follows a predefined disassembly plan. It displays this plan to the operator, monitors the current state of the cylinder, and provides notifications regarding the progress of the operation. At the same time, the system ensures that the collaborative robot operates safely alongside the human operator by synchronizing tasks and monitoring gestures.

Regarding the **modeling and simulation phase**, the robotic cell was designed using CAD tools and integrated into a simulation environment. A disassembly scenario performed by the operator was implemented as an initial demonstration. Due to certain technical limitations related to the software environment, the current simulation scenario only includes the removal of two screws. Nevertheless, this represents a promising starting point and demonstrates the viability of the collaborative workflow.

These results are consistent with findings in collaborative robotics research, which emphasize the importance of combining perception, interaction, and task planning for effective human–robot cooperation.

Finally, I would like to express my sincere appreciation to all members of the team for their dedication and efforts throughout this project. This work was truly the result of effective teamwork, where each member contributed to different aspects of the system.

I personally worked on the overall architecture of the project, including the development of the cylinder traceability system, which tracks the disassembly steps and the detected components using a **YOLO** segmentation model trained specifically for the cylinder. I was also responsible for creating the dataset used to train the model. In addition, I developed the synchronization state machine between the traceability program and the operator gesture detection module, and contributed to the camera calibration process.

I would also like to thank **Hamza Tahri** for his work on the operator gesture detection program, and **Semoug Mouad** for developing the camera calibration program.

Finally, I would like to acknowledge the modeling team — **Sebti Douae** and **Ramdani Souha** — for their work on the CAD design of the robot and the simulation of the disassembly scenario using **3DEXPERIENCE**. Their contributions were essential to the successful completion of this collaborative project.

## Installation

Follow these steps to install the required dependencies and run the project.

### 1. Clone the repository

```bash
git clone https://github.com/he374/Human-robot-Collaborative-desassembly.git
cd Human-robot-Collaborative-desassembly

@duaS
@douad-lgtm
