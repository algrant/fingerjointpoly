import React, { useRef, useState, useMemo } from 'react'
import { Canvas, useFrame, extend, useThree } from '@react-three/fiber'
import * as THREE from 'three';

import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

extend({ OrbitControls })

const Face = ({ points }) => {
  const ref = useRef()

  const lineGeometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points), [points]);

  // const shape = new THREE.Shape();
  
  // shape.moveTo( points[points.length -1].x, points[points.length -1].y, points[points.length -1].z );

  // for(let i = 0; i < points.length; i++) {
  //   shape.lineTo(points[i].x, points[i].y, points[i].z);
  // }

  // const color = 'orange';
  // const mat = new THREE.MeshBasicMaterial({ color, opacity: 0.5, transparent: true });
  // const geom = new THREE.ShapeGeometry(shape);
  // geom.computeBoundingBox()
  // const mesh = new THREE.Mesh(geom, mat);
  // <primitive object={mesh} />

  return <group position={[0,0,0]}>
      <line ref={ref} geometry={lineGeometry}>
        <lineBasicMaterial attach="material" color={'#9c88ff'} linewidth={1} />
      </line>
    </group>
}


const Polyhedron = ({ vertices, faces, triFaces }) => {

    const vFaces = useMemo(() => faces.map(
      face => {
        const vFace = face.map( 
          vId => new THREE.Vector3(vertices[vId][0], vertices[vId][1], vertices[vId][2])
        );
        // make full loop of polygon...
        vFace.push(new THREE.Vector3(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2]));

        return vFace;
      } ) ,[vertices, faces]);

    return <>
      { vFaces.map((face, fId) => <Face key={fId} points={face} /> )}
    </>
}

const CameraControls = () => {
  // stolen directly from here... not pretty, but works

  // Get a reference to the Three.js Camera, and the canvas html element.
  // We need these to setup the OrbitControls component.
  // https://threejs.org/docs/#examples/en/controls/OrbitControls
  const {
    camera,
    gl: { domElement },
  } = useThree();
  // Ref to the controls, so that we can update them on every frame using useFrame
  const controls = useRef();
  useFrame((state) => controls.current.update());
  return <orbitControls ref={controls} args={[camera, domElement]} />;
};


export default function Viewer({ data }) { 
  console.log(data)
  return <div style={{width: '100vw', height: '100vw'}}>
      <Canvas>
      {/* <orbitControls enableDamping /> */}
      <CameraControls />
      <ambientLight />
      <pointLight position={[10, 10, 10]} />
      {/* <Line /> */}
      { data?.model?.data && 
        
        <Polyhedron { ...data.model.data }/>}
    </Canvas>
    </div>;
}