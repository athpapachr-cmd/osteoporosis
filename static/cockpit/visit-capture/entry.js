'use strict';
(() => {const source=new URL(location.href),target=new URL('/static/cockpit/index.html',location.origin);target.searchParams.set('capture','1');const id=source.searchParams.get('patient_id');if(id&&id.length<=120)target.searchParams.set('entry_patient',id);location.replace(target.href);})();
