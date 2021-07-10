import './App.css';
import JSONInput from 'react-json-editor-ajrm';
import locale    from 'react-json-editor-ajrm/locale/en';

import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [data, setData] = useState({ hits: [] });
  const init = { model: 1};
  const [query, setQuery] = useState({ model: 1 });

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
          // height      = '550px'
          viewOnly
      />
    </div>
  );
}

export default App;
