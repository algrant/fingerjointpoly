import './App.css';
import JSONInput from 'react-json-editor-ajrm';
import locale    from 'react-json-editor-ajrm/locale/en';

import React, { useState, useEffect } from 'react';
import axios from 'axios';

import Viewer from './fjpViewer.js';

function App() {
  const [data, setData] = useState({});
  const init = { model: 95 };
  const [query, setQuery] = useState({ model: 95 });

  useEffect(() => {
    const update = async () => {
      const result = await axios(
        `http://127.0.0.1:8000/models/${query.model}`,
      );
      setData(result.data);
    }
    update();
  }, [query]);

  const test = (change) => {
    setQuery(change.jsObject)
    setData({ loading: query});
  }

  return (
    <div className="FJP">
      <div>Input</div>
      <JSONInput
        placeholder={ init }
        onChange={test}
        height = '150px'
      />
      <div> Output</div>
      <JSONInput
          id          = 'data_view'
          placeholder = { data }
          locale      = { locale }
          height      = '250px'
          viewOnly
      />
      <Viewer data={data} />
    </div>
  );
}

export default App;
