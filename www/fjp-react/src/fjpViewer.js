import React, { useRef, useState, useMemo, useUpdate } from 'react'
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
        <lineBasicMaterial attach="material" color={'#FFF'} linewidth={1} />
      </line>
    </group>
}


const Polyhedron = ({ vertices, faces, triangles }) => {

    const vFaces = useMemo(() => faces.map(
      face => {
        const vFace = face.map( 
          vId => new THREE.Vector3(vertices[vId][0], vertices[vId][1], vertices[vId][2])
        );
        // make full loop of polygon...
        vFace.push(new THREE.Vector3(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2]));

        return vFace;
      } ) ,[vertices, faces]);

    const vTriangles = useMemo(() => triangles.map((v) => new THREE.Vector3(v[0], v[1], v[2])), [triangles]);
    console.log(vTriangles)

    // const radius = 0.1;
    // const detail = 2;
    // {vTriangles && <polyhedronBufferGeometry
    //   attach="geometry"
    //   args={[vTriangles, [...Array(vTriangles.length).keys()], radius, detail]}
    // />}

    //works but is scaled annoyingly...
    // radius = 1
    // detail = 0
    // var geometry = new THREE.PolyhedronGeometry( triangles.flat(), [...Array(vTriangles.length).keys()], radius, detail);

    const geometry = new THREE.BufferGeometry();
    // create a simple square shape. We duplicate the top left and bottom right
    // vertices because each vertex needs to appear once per triangle.
    const verts2 = useMemo(() => new Float32Array( triangles.flat() ), [triangles]); //vertices.slice(0, 6).flat());
    console.log(verts2)
    // itemSize = 3 because there are 3 values (components) per vertex
    geometry.setAttribute( 'position', new THREE.BufferAttribute( verts2, 3 ) );

    return <>
      { vFaces.map((face, fId) => <Face key={fId} points={face} /> )}
      <mesh 
        onClick={e => console.log('click')} 
        onPointerOver={e => console.log('hover')} 
        onPointerOut={e => console.log('unhover')}
        geometry={geometry}
        >
        {/* <boxBufferGeometry args={[1,1,1]} /> */}

        <meshPhongMaterial attach="material" color="red" opacity={1} flatShading reflectivity={0}/>
      </mesh>
    </>
}

const CameraControls = () => {
  // stolen directly from somewhere... not pretty, but works

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
      <pointLight position={[20, 20, 20]} />
      {/* <Line /> */}
      { data?.model?.data && 
        
        <Polyhedron { ...data.model.data }/>}
    </Canvas>
    </div>;
}