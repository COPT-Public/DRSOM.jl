-- A clean script for `*.result`
-- for (a single problem name, param, method)
-- only keeps first several versions
delete cutest.result from cutest.result join (SELECT m.*, ROW_NUMBER() OVER (PARTITION BY name,param,method ORDER BY `update` DESC) AS rn
  FROM cutest.result AS m) tt
on tt.id = cutest.result.id
where rn >= 2;