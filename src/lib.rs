use csv::{Reader, StringRecord, Writer};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

// Comp enum for filtering
pub enum Comp {
    Eq,
    Geq,
    Leq,
    Les,
    Gra,
    Not,
}

pub fn compare<T: PartialOrd>(a: T, op: &Comp, b: T) -> bool {
    match op {
        Comp::Eq => a.eq(&b),
        Comp::Les => a < b,
        Comp::Gra => a > b,
        Comp::Geq => a.ge(&b),
        Comp::Leq => a.le(&b),
        Comp::Not => a != b,
    }
}

// Column trait for general columns
#[derive(Clone)]
pub enum Column<T> {
    Numeric(NumericColumn<T>),
    Discrete(DiscreteColumn),
}

impl<
        T: Clone
            + Eq
            + std::hash::Hash
            + Add
            + Div
            + Mul
            + Sub
            + core::cmp::PartialOrd
            + std::string::ToString,
    > Column<T>
{
    fn get_key(&self) -> &String {
        match self {
            Self::Discrete(x) => &x.key,
            Self::Numeric(x) => &x.key,
        }
    }

    fn get_num(&self, index: usize) -> Option<&T> {
        match self {
            Self::Discrete(_) => None,
            Self::Numeric(n) => Some(n.get(index)),
        }
    }

    fn filter_array(&self, comp: Comp, val: Option<T>, str_val: Option<String>) -> Vec<bool> {
        match self {
            Column::Discrete(d) => d.filter_array(&str_val.unwrap()),
            Column::Numeric(n) => n.filter_array(&val.unwrap(), comp),
        }
    }

    fn binary_view(&self, picker: &[bool]) -> Column<T> {
        match self {
            Column::Numeric(n) => Column::Numeric(n.binary_view(picker)),
            Column::Discrete(d) => Column::Discrete(d.binary_view(picker)),
        }
    }

    fn len(&self) -> usize {
        match self {
            Column::Numeric(n) => n.len(),
            Column::Discrete(d) => d.len(),
        }
    }

    fn to_string(&self) -> String {
        match self {
            Column::Numeric(n) => n.to_string(),
            Column::Discrete(d) => d.to_string(),
        }
    }
}

// DiscreteColumn struct contains only string values
#[derive(Clone)]
pub struct DiscreteColumn {
    key: String,
    items: Vec<String>,
}

impl DiscreteColumn {
    // Take a binary view of the numeric column, true values are preserved, false values are ignored
    pub fn binary_view(&self, picker: &[bool]) -> DiscreteColumn {
        DiscreteColumn {
            key: self.key.clone(),
            items: self
                .items
                .iter()
                .zip(picker.iter())
                .filter(|(_, b)| **b)
                .map(|(a, _)| a.clone())
                .collect(),
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> DiscreteColumn {
        DiscreteColumn {
            key: self.key.clone(),
            items: self.items[start..end].to_vec(),
        }
    }

    pub fn values(&self) -> HashSet<String> {
        self.items.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn get(&self, index: usize) -> &String {
        &self.items[index]
    }

    pub fn filter_array(&self, val: &String) -> Vec<bool> {
        let mut filter = Vec::new();
        for n in self.items.iter() {
            if val.eq(n) {
                filter.push(true)
            } else {
                filter.push(false)
            }
        }
        filter
    }

    pub fn to_string(&self) -> String {
        let mut result = self.key.clone();
        let data = self.items.join(", ");
        result.push_str(": [");
        result.push_str(&data);
        result.push_str("]");
        result
    }
}

// NumericColumn struct is roughly equivalent to pandas Series
#[derive(Clone)]
pub struct NumericColumn<T> {
    key: String,
    items: Vec<T>,
}

impl<
        T: Clone
            + Eq
            + std::hash::Hash
            + Add
            + Div
            + Mul
            + Sub
            + core::cmp::PartialOrd
            + std::string::ToString,
    > NumericColumn<T>
{
    // Take a binary view of the numeric column, true values are preserved, false values are ignored
    pub fn binary_view(&self, picker: &[bool]) -> NumericColumn<T> {
        NumericColumn {
            key: self.key.clone(),
            items: self
                .items
                .iter()
                .zip(picker.iter())
                .filter(|(_, b)| **b)
                .map(|(a, _)| a.clone())
                .collect(),
        }
    }

    pub fn slice(&self, start: usize, end: usize) -> NumericColumn<T> {
        NumericColumn {
            key: self.key.clone(),
            items: self.items[start..end].to_vec(),
        }
    }

    pub fn values(&self) -> HashSet<T> {
        self.items.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn get(&self, index: usize) -> &T {
        &self.items[index]
    }

    pub fn filter_array(&self, val: &T, comparison: Comp) -> Vec<bool> {
        self.items
            .iter()
            .map(|x| compare(x, &comparison, val))
            .collect()
    }

    pub fn to_string(&self) -> String {
        let mut result = self.key.clone();
        let str_form: Vec<String> = self.items.iter().map(|x| x.to_string()).collect();
        result.push_str(": [");
        result.push_str(&str_form.join(", "));
        result.push_str("]");

        result
    }
}

// Build function for building a numeric column
pub fn build_column_numeric<T>(key: &str, data: Vec<T>) -> Column<T> {
    Column::Numeric(NumericColumn {
        key: String::from(key),
        items: data,
    })
}

// Build function for building a discrete (String) column
pub fn build_column_discrete<T>(key: &str, data: Vec<String>) -> Column<T> {
    Column::Discrete(DiscreteColumn {
        key: String::from(key),
        items: data,
    })
}

#[derive(Clone)]
pub struct NodFrame<T> {
    columns: Vec<Column<T>>,
    column_idx: HashMap<String, usize>,
    num_rows: usize,
    num_cols: usize,
}

impl<
        T: Clone + Eq + std::hash::Hash + Add + Div + Mul + Sub + PartialOrd + std::string::ToString,
    > NodFrame<T>
{
    // numeric_cols returns the column names of numeric columns
    pub fn numeric_cols(&self) -> Vec<&String> {
        let mut num_col = Vec::new();
        for column in self.columns.iter() {
            if let Column::Numeric(a) = column {
                num_col.push(&a.key)
            }
        }
        num_col
    }

    // numeric_index returns the column indices of number columns
    pub fn numeric_index(&self) -> Vec<usize> {
        let cols = self.numeric_cols();
        let mut indices = Vec::new();
        for key in cols {
            indices.push(self.column_idx.get(key).unwrap().clone())
        }
        indices
    }

    // numeric_rows returns the rows of numeric columns as a 2d matrix
    pub fn numeric_rows(&self) -> Vec<Vec<T>> {
        let numeric_index = self.numeric_index();
        let mut data = Vec::new();
        for i in 0..self.num_rows {
            let mut row = Vec::new();
            for col in numeric_index.iter() {
                row.push(self.columns[*col].get_num(i).unwrap().clone())
            }
            data.push(row)
        }
        data
    }

    pub fn filter_frame(
        &self,
        col: String,
        comp: Comp,
        val: Option<T>,
        str_val: Option<String>,
    ) -> NodFrame<T> {
        let col_idx = self.column_idx.get(&col).unwrap().clone();
        let picker = self.columns[col_idx].filter_array(comp, val, str_val);
        let mut copy = self.clone();
        copy.columns = copy
            .columns
            .iter()
            .map(|x| x.binary_view(&picker))
            .collect();
        copy.num_rows = copy.columns[0].len();
        copy
    }

    pub fn to_csv(&self, file_path: String) -> Result<(), Box<dyn Error>> {
        let file = File::create(file_path)?;
        let mut writer = Writer::from_writer(file);
        writer.write_record(self.columns.iter().map(|x| x.get_key()))?;
        for i in 0..self.num_rows {
            let mut row = Vec::new();
            for col in self.columns.iter() {
                match col {
                    Column::Numeric(n) => {
                        row.push(n.get(i).to_string());
                    }
                    Column::Discrete(d) => {
                        row.push(d.get(i).clone());
                    }
                }
            }
            writer.write_record(&row)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn to_string(&self) -> String {
        let mut result = String::from("nodframe:\n");
        for col in self.columns.iter() {
            result.push_str(&col.to_string());
            result.push_str("\n");
        }
        result.push_str("Num Rows: ");
        result.push_str(&self.num_rows.to_string());

        result
    }
}

// frame_from_csv reads in a csv and automatically converts it into a
pub fn frame_from_csv<
    T: Clone
        + Eq
        + std::hash::Hash
        + Add
        + Div
        + Mul
        + Sub
        + PartialOrd
        + std::string::ToString
        + FromStr,
>(
    file_path: String,
) -> Result<NodFrame<T>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = Reader::from_reader(file);
    let mut record = StringRecord::new();
    let head = reader.headers()?.clone();
    let header: Vec<&str> = head.iter().collect();
    let mut data: Vec<Vec<String>> = vec![vec![]; header.len()];

    while !reader.is_done() {
        let row: Vec<String>;
        if reader.read_record(&mut record)? {
            row = record.iter().map(|s| s.to_string()).collect();
        } else {
            break;
        }
        for i in 0..header.len() {
            data[i].push(row[i].clone());
        }
    }
    let mut num_keys = Vec::new();
    let mut disc_keys = Vec::new();
    let mut num_data = Vec::new();
    let mut disc_data = Vec::new();

    for i in 0..header.len() {
        if let Ok(_) = data[i][0].parse::<T>() {
            num_keys.push(header[i].to_string());
            let mut col = Vec::new();
            for element in data[i].iter() {
                if let Ok(n) = element.parse::<T>() {
                    col.push(n);
                }
            }
            num_data.push(col);
        } else {
            disc_keys.push(header[i].to_string());
            disc_data.push(data[i].clone());
        }
    }
    Ok(frame_from_vecs(num_keys, num_data, disc_keys, disc_data))
}

// Build functions for Frame
pub fn frame_from_vecs<
    T: Clone + Eq + std::hash::Hash + Add + Div + Mul + Sub + PartialOrd + std::string::ToString,
>(
    num_keys: Vec<String>,
    num_data: Vec<Vec<T>>,
    str_keys: Vec<String>,
    str_data: Vec<Vec<String>>,
) -> NodFrame<T> {
    let data_rows = num_data[0].len();

    let num_columns: Vec<Column<T>> = num_keys
        .iter()
        .zip(num_data.iter())
        .map(|(k, v)| {
            Column::Numeric(NumericColumn {
                key: k.to_string(),
                items: v.to_vec(),
            })
        })
        .collect();

    let str_columns: Vec<Column<T>> = str_keys
        .iter()
        .zip(str_data)
        .map(|(k, v)| {
            Column::Discrete(DiscreteColumn {
                key: k.to_string(),
                items: v.to_vec(),
            })
        })
        .collect();

    let cols: Vec<Column<T>> = num_columns
        .into_iter()
        .chain(str_columns.into_iter())
        .collect();

    let names: HashMap<String, usize> = (0..cols.len())
        .map(|i| (cols[i].get_key().clone(), i))
        .collect();

    NodFrame {
        num_cols: cols.len(),
        columns: cols,
        column_idx: names,
        num_rows: data_rows,
    }
}

///// TESTS /////

#[cfg(test)]
mod col_tests {
    use super::*;

    #[test]
    fn binary_view_test() {
        let col = NumericColumn {
            key: String::from("bing"),
            items: vec![1, 2, 3],
        };
        let b = col.binary_view(&vec![true, false, true]);
        assert_eq!(vec![1, 3], b.items)
    }

    #[test]
    fn slice_test() {
        let col = NumericColumn {
            key: String::from("bing"),
            items: vec![1, 2, 3],
        };
        let b = col.slice(1, 3);
        assert_eq!(vec![2, 3], b.items)
    }

    #[test]
    fn values_test() {
        let col = NumericColumn {
            key: String::from("bing"),
            items: vec![1_i64, 2, 3, 3, 2, 1, 4],
        };
        let b = col.values();
        let mut c = HashSet::new();
        c.extend(vec![1, 2, 3, 4].iter());
        assert_eq!(b, c);
    }

    #[test]
    fn filter_test() {
        let col = NumericColumn {
            key: String::from("bing"),
            items: vec![1_i64, 2, 3, 3, 2, 1, 4],
        };
        let b = col.filter_array(&2, Comp::Leq);
        let c = vec![true, true, false, false, true, true, false];
        assert_eq!(c, b);
    }
}

mod frame_tests {
    use super::*;

    #[test]
    fn csv_test() {
        let frame = frame_from_vecs(
            vec![String::from("value"), String::from("whoop")],
            vec![vec![1, 2, 3, 4], vec![14, 54, 7, 2]],
            vec![String::from("kabang")],
            vec![vec![
                String::from("1a"),
                String::from("2a"),
                String::from("3a"),
                String::from("4a"),
            ]],
        );
        frame.to_csv(String::from("hehe.csv"));
        let frame2 = frame_from_csv::<i32>(String::from("hehe.csv")).unwrap();
        assert_eq!(frame.to_string(), frame2.to_string());
    }
}
